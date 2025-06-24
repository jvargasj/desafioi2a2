import requests
import regex as re

def fetch_model():
    url = "http://localhost:1234/v1/models"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        model_id = data['data'][0]['id']
        match = re.search(r"/([^/]+)\.gguf$", model_id)
        if match:
            result = match.group(1)
            return result
        else:
            raise ValueError("No match found in the model ID.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")

model = fetch_model()