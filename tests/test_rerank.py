import requests

payload = {
        "query": "What is the capital of France?",
        "max_length": 128,
        "model": "pairwise_small",
        "passages": [
            {
            "id": 1,
            "text": "France is a country in Western Europe known for its rich history, cuisine, and cultural influence. The country has several major cities including Marseille, Lyon, and Paris."
            },
            {
            "id": 2,
            "text": "Paris is the capital and most populous city of France. It is the political, economic, and cultural center of the country."
            },
            {
            "id": 3,
            "text": "Lyon is the gastronomic capital and a major city in France known for its culinary traditions and central location."
            }
        ]
    }

r = requests.post("http://localhost:8181/rerank", json=payload, timeout=60)
if r.status_code == 422:
    print("422 details:", r.json())
r.raise_for_status()

data = r.json()

for i, p in enumerate(data["results"], 1):
    print(i, p["id"], p["score"], p["text"][:60])
