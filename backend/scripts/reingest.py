import csv, httpx

with open('backend/data/metrics.csv') as f:
    rows = list(csv.DictReader(f))[:100]

for row in rows:
    r = httpx.post(
        'http://localhost:8000/metrics/ingest',
        json={
            'cpu_percent': float(row['cpu_percent']),
            'ram_percent': float(row['ram_percent']),
            'latency_ms': float(row['latency_ms']),
            'error_rate': float(row['error_rate']),
        }
    )
    print(r.status_code)

print('Done — check /metrics/ml/status')