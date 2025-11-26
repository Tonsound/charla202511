import json
import requests

# -------------------------------------------------------
# 1. INSERT YOUR API GATEWAY INVOKE URL HERE
# -------------------------------------------------------
API_URL = "https://1g3vuy92cc.execute-api.us-east-1.amazonaws.com/prod/next-best-product"


# -------------------------------------------------------
# 2. Example test payload (matches your feature schema)
# -------------------------------------------------------
payload = {
    "last_purchase": 12,  # days since last purchase
    "frequency": 18,
    "monetary": 240.75,
    "unique_products": 7,
    "total_quantity": 29,
    "recency": 12,
    "avg_order_value": 13.37,
    "product_popularity": 410,
    "product_avg_price": 2.55
}


# -------------------------------------------------------
# 3. Send request
# -------------------------------------------------------
def test_api():
    print("➡️ Sending request to API Gateway...")
    print("POST", API_URL)
    print("Payload:\n", json.dumps(payload, indent=2))

    response = requests.post(
        API_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )

    print("\n⬅️ Response status:", response.status_code)

    try:
        body = response.json()
        print("\nResponse JSON:")
        print(json.dumps(body, indent=2))
    except Exception:
        print("\nRaw response:")
        print(response.text)


if __name__ == "__main__":
    test_api()
