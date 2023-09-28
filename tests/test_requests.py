import requests
import json

response = requests.post("http://0.0.0.0:8001/classify", json={"filepath": "/path/to/image.jpg"})
print(response.json())

# python test_requests.py