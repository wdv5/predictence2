"""
backend/scripts/bootstrap_prophet_data.py
==========================================
Generates synthetic historical metric rows and optionally re-ingests them
into the Predictence backend so the Isolation Forest trains immediately.

Usage
-----
# Step 1 — generate CSV only
python backend/scripts/bootstrap_prophet_data.py \\
    --out backend/data/metrics.csv \\
    --points 240

# Step 2 — generate + auto-ingest in one command
python backend/scripts/bootstrap_prophet_data.py \\
    --out backend/data/metrics.csv \\
    --points 240 \\
    --ingest \\
    --url http://localhost:8000 \\
    --ingest-limit 100

# Generate with anomaly injection so the IF model learns a richer boundary
python backend/scripts/bootstrap_prophet_data.py \\
    --out backend/data/metrics.csv \\
    --points 480 \\
    --anomaly-rate 0.05 \\
    --ingest \\
    --ingest-limit 200
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap")

# ---------------------------------------------------------------------------
# Field names must match backend/core/predictive_monitor.py _FIELDNAMES
# ---------------------------------------------------------------------------
FIELDNAMES = ["timestamp", "cpu_percent", "ram_percent", "latency_ms", "error_rate", "source"]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def generate_normal_row(t: float, rng: random.Random) -> dict:
    """
    Realistic 'healthy' metrics with sinusoidal diurnal patterns and
    Gaussian noise.  `t` is a fractional hour from epoch (0 = midnight).
    """
    # Diurnal CPU: peaks mid-morning and mid-afternoon
    diurnal = (
        12 * math.sin(2 * math.pi * (t - 9) / 24)   # morning peak
        + 8 * math.sin(2 * math.pi * (t - 14) / 24) # afternoon secondary
    )
    cpu = _clamp(42 + diurnal + rng.gauss(0, 4), 5, 78)

    # RAM slowly follows CPU with a lag
    ram = _clamp(52 + 0.3 * diurnal + rng.gauss(0, 3), 30, 80)

    # Latency loosely correlated with CPU
    lat = _clamp(110 + 1.2 * (cpu - 42) + rng.gauss(0, 20), 30, 480)

    # Error rate near-zero during normal operation
    err = _clamp(0.4 + rng.uniform(0, 1.2), 0, 4)

    return {"cpu_percent": cpu, "ram_percent": ram, "latency_ms": lat, "error_rate": err}


def generate_anomaly_row(kind: str, rng: random.Random) -> dict:
    """
    Inject one of four anomaly patterns so the model learns
    a richer decision boundary.
    """
    if kind == "cpu_spike":
        return {
            "cpu_percent": _clamp(rng.gauss(88, 4), 80, 100),
            "ram_percent": _clamp(rng.gauss(70, 4), 60, 85),
            "latency_ms":  _clamp(rng.gauss(380, 60), 200, 700),
            "error_rate":  _clamp(rng.uniform(1.5, 4), 0, 10),
        }
    if kind == "latency":
        return {
            "cpu_percent": _clamp(rng.gauss(56, 6), 35, 75),
            "ram_percent": _clamp(rng.gauss(62, 4), 50, 78),
            "latency_ms":  _clamp(rng.gauss(700, 120), 500, 1200),
            "error_rate":  _clamp(rng.uniform(4, 10), 0, 20),
        }
    if kind == "ram_pressure":
        return {
            "cpu_percent": _clamp(rng.gauss(62, 5), 40, 80),
            "ram_percent": _clamp(rng.gauss(91, 3), 85, 99),
            "latency_ms":  _clamp(rng.gauss(280, 50), 100, 600),
            "error_rate":  _clamp(rng.uniform(0.5, 3), 0, 8),
        }
    # cascade
    return {
        "cpu_percent": _clamp(rng.gauss(93, 3), 85, 100),
        "ram_percent": _clamp(rng.gauss(94, 2), 88, 99),
        "latency_ms":  _clamp(rng.gauss(1150, 150), 800, 1800),
        "error_rate":  _clamp(rng.uniform(10, 25), 0, 30),
    }


ANOMALY_KINDS = ["cpu_spike", "latency", "ram_pressure", "cascade"]


def generate_rows(
    n_points: int,
    anomaly_rate: float,
    start: Optional[datetime],
    interval_seconds: int,
    seed: int,
) -> list[dict]:
    """
    Build a list of metric dicts with realistic timestamps.
    `anomaly_rate` fraction of rows will be injected anomalies.
    """
    rng = random.Random(seed)
    rows: list[dict] = []

    if start is None:
        # Default: end at 'now', look back n_points * interval
        start = datetime.utcnow() - timedelta(seconds=n_points * interval_seconds)

    for i in range(n_points):
        ts = start + timedelta(seconds=i * interval_seconds)
        hour_of_day = ts.hour + ts.minute / 60.0

        if rng.random() < anomaly_rate:
            kind = rng.choice(ANOMALY_KINDS)
            metrics = generate_anomaly_row(kind, rng)
            source = f"bootstrap_anomaly_{kind}"
        else:
            metrics = generate_normal_row(hour_of_day, rng)
            source = "bootstrap_normal"

        rows.append(
            {
                "timestamp": ts.isoformat(),
                "source": source,
                **metrics,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows → %s", len(rows), out_path)


# ---------------------------------------------------------------------------
# Ingest via HTTP
# ---------------------------------------------------------------------------

def ingest_rows(
    rows: list[dict],
    base_url: str,
    limit: int,
    delay: float,
    dry_run: bool,
) -> None:
    try:
        import httpx
    except ImportError:
        log.error("httpx is not installed. Run: pip install httpx")
        sys.exit(1)

    rows_to_send = rows[:limit]
    log.info(
        "Ingesting %d rows into %s/metrics/ingest (delay=%.2fs, dry_run=%s)",
        len(rows_to_send),
        base_url,
        delay,
        dry_run,
    )

    url = f"{base_url.rstrip('/')}/metrics/ingest"
    succeeded = 0
    failed = 0

    for idx, row in enumerate(rows_to_send, 1):
        payload = {
            "cpu_percent": float(row["cpu_percent"]),
            "ram_percent": float(row["ram_percent"]),
            "latency_ms":  float(row["latency_ms"]),
            "error_rate":  float(row["error_rate"]),
        }

        if dry_run:
            log.debug("[dry-run] %d/%d  %s", idx, len(rows_to_send), payload)
            succeeded += 1
            continue

        try:
            resp = httpx.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            succeeded += 1
        except httpx.HTTPStatusError as exc:
            log.warning("Row %d — HTTP %s: %s", idx, exc.response.status_code, exc.response.text[:80])
            failed += 1
        except Exception as exc:
            log.warning("Row %d — request failed: %s", idx, exc)
            failed += 1

        # Progress heartbeat every 25 rows
        if idx % 25 == 0:
            log.info("  … %d/%d sent (%d ok, %d failed)", idx, len(rows_to_send), succeeded, failed)

        if delay > 0:
            time.sleep(delay)

    log.info("Ingest complete — %d succeeded, %d failed", succeeded, failed)

    # Poll /metrics/ml/status to show training progress
    if not dry_run:
        _print_ml_status(base_url)


def _print_ml_status(base_url: str) -> None:
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/metrics/ml/status", timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            trained = d.get("trained", False)
            samples = d.get("training_samples", "?")
            needed  = d.get("min_samples_needed", 50)
            mode    = d.get("mode", "?")
            log.info(
                "ML status → trained=%s | samples=%s/%s | mode=%s",
                trained, samples, needed, mode,
            )
            if trained:
                log.info("✓ Isolation Forest is ACTIVE — anomaly detection online.")
            else:
                remaining = max(0, needed - (samples if isinstance(samples, int) else 0))
                log.info(
                    "⏳ Model warming up — ingest %d more samples to activate.", remaining
                )
    except Exception as exc:
        log.debug("Could not fetch ML status: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic historical metrics for Predictence and optionally ingest them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Generation
    p.add_argument(
        "--out",
        default="backend/data/metrics.csv",
        help="Output CSV path (default: backend/data/metrics.csv)",
    )
    p.add_argument(
        "--points",
        type=int,
        default=240,
        help="Number of metric rows to generate (default: 240)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between synthetic samples (default: 300 = 5 min)",
    )
    p.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        metavar="RATE",
        help="Fraction of rows to inject as anomalies, 0.0–1.0 (default: 0.05)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--start",
        default=None,
        metavar="ISO_DATETIME",
        help="Start timestamp in ISO format, e.g. 2024-01-01T00:00:00 (default: derived from --points)",
    )

    # Ingest
    p.add_argument(
        "--ingest",
        action="store_true",
        help="After generating, POST each row to /metrics/ingest",
    )
    p.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Predictence backend base URL (default: http://localhost:8000)",
    )
    p.add_argument(
        "--ingest-limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum rows to ingest (default: 100, enough to train the IF model)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Delay between ingest requests in seconds (default: 0 = as fast as possible)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate but do not actually send HTTP requests",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate
    if args.points < 1:
        parser.error("--points must be >= 1")
    if not (0.0 <= args.anomaly_rate <= 1.0):
        parser.error("--anomaly-rate must be between 0.0 and 1.0")
    if args.ingest and args.ingest_limit < 1:
        parser.error("--ingest-limit must be >= 1")

    start_dt: Optional[datetime] = None
    if args.start:
        try:
            start_dt = datetime.fromisoformat(args.start)
        except ValueError:
            parser.error(f"--start '{args.start}' is not a valid ISO datetime")

    log.info(
        "Generating %d rows | interval=%ds | anomaly_rate=%.0f%% | seed=%d",
        args.points,
        args.interval,
        args.anomaly_rate * 100,
        args.seed,
    )

    rows = generate_rows(
        n_points=args.points,
        anomaly_rate=args.anomaly_rate,
        start=start_dt,
        interval_seconds=args.interval,
        seed=args.seed,
    )

    out_path = Path(args.out)
    write_csv(rows, out_path)

    # Summary stats
    cpus  = [r["cpu_percent"] for r in rows]
    rams  = [r["ram_percent"] for r in rows]
    lats  = [r["latency_ms"]  for r in rows]
    errs  = [r["error_rate"]  for r in rows]
    n_anom = sum(1 for r in rows if "anomaly" in r["source"])

    log.info(
        "Stats — CPU: %.1f–%.1f%%  RAM: %.1f–%.1f%%  "
        "Latency: %.0f–%.0fms  Errors: %.1f–%.1f%%  Anomalies injected: %d",
        min(cpus), max(cpus),
        min(rams), max(rams),
        min(lats), max(lats),
        min(errs), max(errs),
        n_anom,
    )

    if args.ingest:
        ingest_rows(
            rows=rows,
            base_url=args.url,
            limit=args.ingest_limit,
            delay=args.delay,
            dry_run=args.dry_run,
        )
    else:
        log.info(
            "Tip: re-run with --ingest --ingest-limit %d to train the Isolation Forest immediately.",
            min(args.points, 100),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
