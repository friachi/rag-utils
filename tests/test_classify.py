import requests, json

payload = {
    "text": "root: fpml:trade elements: swap fixedRateSchedule floatingRateIndex notional paymentDate",
    "labels": ["Interest Rate Swap", "FX Forward", "Bond", "Equity Option", "Credit Default Swap"],
    "multi_label": False,
    "top_k": 3
  }

r = requests.post("http://localhost:8181/classify", json=payload, timeout=60)
if r.status_code == 422:
    print("422 details:", r.json())
r.raise_for_status()

data = r.json()


print(json.dumps(data, indent=2))

