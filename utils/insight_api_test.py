# insight_api_test.py
import requests
import json

base_url = "http://localhost:8004"
test_url = f"{base_url}/search_get"  # Changed to match the new GET endpoint
test_params = {
    "query": "test query",
    "num_results": 1
}

try:
    print(f"Sending GET request to: {test_url}")
    print(f"Request params: {json.dumps(test_params, indent=2)}")

    response = requests.get(test_url, params=test_params, timeout=10)  # Added timeout

    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
    print(f"Response content: {response.text}")

    response.raise_for_status()
    print("Connection successful!")
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Connection failed: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Error response content: {e.response.text}")
except json.JSONDecodeError:
    print("Failed to decode JSON response")
