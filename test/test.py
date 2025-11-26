import requests
import json

# API Gateway URL
API_URL = "https://1g3vuy92cc.execute-api.us-east-1.amazonaws.com/prod/next-best-product"

# Test SKUs to check recommendations
test_skus = ["71053"]

for sku in test_skus:
    payload = {"sku": sku}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(data)
            print(f"\nInput SKU: {data['input_sku']}")
            print(f"Recommended product ID: {data['recommended_product_id']}")
            print(f"Recommended product name: {data['recommended_product_name']}")
            print(f"Score: {data['score']:.4f}")
            print(f"Marketing Message: {data['marketing_message']}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception for SKU {sku}: {e}")
