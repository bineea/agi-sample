import requests
import os
import uuid

api_key = 'sk-BxgzdB8v7YfD50PGF_y_GJMhI9ow7A85r6Kdxyt0ZqQ'

# API endpoint configuration
# url = "http://localhost:7860/api/v1/run/e04c0982-fc6c-4022-beae-9e8453c98e92"

# Using webhook endpoint as per the provided code context
url = "http://localhost:7860/api/v1/webhook/e04c0982-fc6c-4022-beae-9e8453c98e92"

# Request payload configuration
payload = {
    "output_type": "text",
    "input_type": "text",
    "input_value": "hello world!"
}
payload["session_id"] = str(uuid.uuid4())
payload["test_value"] = 'test'


headers = {"x-api-key": api_key}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")