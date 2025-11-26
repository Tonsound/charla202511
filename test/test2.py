import requests
import json
import random

API_URL = "https://1g3vuy92cc.execute-api.us-east-1.amazonaws.com/prod/next-best-product"

# Candidate products (only numeric)
candidate_products = [
    "71053","22752","21730",
    "22633","22632","84879","22745","22748","22749","22310",
    "84969","22623","22622","21754","21755","21777"
]

# Fake product names
product_names = {
    "71053": "Wireless Mouse",
    "22752": "USB-C Hub",
    "21730": "Bluetooth Speaker",
    "22633": "Gaming Keyboard",
    "22632": "Noise-Cancel Headphones",
    "84879": "Smart LED Bulb",
    "22745": "Portable Charger",
    "22748": "Laptop Stand",
    "22749": "Webcam HD",
    "22310": "External SSD",
    "84969": "Smartwatch",
    "22623": "Wireless Earbuds",
    "22622": "Fitness Tracker",
    "21754": "HDMI Cable",
    "21755": "Ethernet Adapter",
    "21777": "Tablet Case"
}

# Generate dummy features for each product
def generate_dummy_features(product_id):
    return {
        "customer_id": random.randint(10000, 20000),
        "product_id": int(product_id),  # convert to numeric for SageMaker XGBoost
        "recency": random.randint(1, 100),
        "frequency": random.randint(1, 50),
        "monetary": round(random.uniform(10, 1000), 2),
        "unique_products": random.randint(1, 20),
        "total_quantity": random.randint(1, 100),
        "avg_order_value": round(random.uniform(10, 500), 2),
        "product_popularity": random.randint(1, 500),
        "product_avg_price": round(random.uniform(5, 300), 2)
    }

results = []

for product_id in candidate_products:
    payload = generate_dummy_features(product_id)
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            prediction = float(data['prediction'])
            results.append((product_id, prediction))
        else:
            print(f"Error calling API for product {product_id}: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Exception for product {product_id}: {e}")

# Sort by score descending
results.sort(key=lambda x: x[1], reverse=True)

print("\nNext-best-product recommendations (product_name, score):")
for product_id, score in results:
    name = product_names.get(product_id, f"Product {product_id}")
    print(name, score)

# Pick the single best recommendation
if results:
    best_product_id, best_score = results[0]
    best_name = product_names.get(best_product_id, f"Product {best_product_id}")
    print(f"\n✅ Best product recommendation: {best_name} ({best_product_id}) with score {best_score}")
else:
    print("\nNo valid product recommendations received.")
