import urllib.request
import json
import random

BASE_URL = "http://localhost:8080"

def test_stats():
    print("Testing GET /stats:")
    req = urllib.request.Request(f"{BASE_URL}/stats")
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Stats: {data}")
        assert data["dimension"] == 64

def test_add():
    print("\nTesting POST /vectors/add:")
    vectors = []
    for i in range(10):
        vectors.append({
            "id": i,
            "vector": [random.random() for _ in range(64)]
        })
    
    payload = json.dumps({"vectors": vectors}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/vectors/add", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Add Response: {data}")
        assert data["status"] == "success"
        assert data["added"] == 10

def test_search():
    print("\nTesting POST /search:")

    query = [0.5 for _ in range(64)]
    
    payload = json.dumps({"vector": query, "k": 3}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/search", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Search Results: {data}")
        assert "results" in data
        assert len(data["results"]) == 3

if __name__ == "__main__":
    try:
        test_stats()
        test_add()
        test_search()
        test_stats()
        print("\nAll REST API tests passed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)
