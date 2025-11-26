import boto3
import os
import pandas as pd
import json
import random

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "online-retail-xgb-serverless")

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

# Candidate products (numeric IDs only)
candidate_products = list(product_names.keys())

# Generate dummy features for a product
def generate_dummy_features(product_id):
    return {
        "customer_id": random.randint(10000, 20000),
        "product_id": int(product_id),
        "recency": random.randint(1, 100),
        "frequency": random.randint(1, 50),
        "monetary": round(random.uniform(10, 1000), 2),
        "unique_products": random.randint(1, 20),
        "total_quantity": random.randint(1, 100),
        "avg_order_value": round(random.uniform(10, 500), 2),
        "product_popularity": random.randint(1, 500),
        "product_avg_price": round(random.uniform(5, 300), 2)
    }

def lambda_handler(event, context):
    try:
        # Extract SKU from the API call
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event

        input_sku = str(body.get("sku"))
        if input_sku not in candidate_products:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"SKU {input_sku} is not in candidate products."})
            }

        # Generate predictions for all candidate products
        results = []
        for product_id in candidate_products:
            payload = generate_dummy_features(product_id)
            # Build CSV for SageMaker
            df = pd.DataFrame([payload])
            csv_payload = df.to_csv(index=False, header=False)

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='text/csv',
                Body=csv_payload
            )

            result = response['Body'].read().decode('utf-8')
            try:
                prediction = float(json.loads(result)) if result.strip().startswith('[') else float(result)
            except Exception:
                prediction = float(result) if isinstance(result, str) else 0.0

            results.append((product_id, prediction))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Pick the top recommendation that is NOT the input SKU
        for best_product_id, best_score in results:
            if best_product_id != input_sku:
                best_name = product_names.get(best_product_id, f"Product {best_product_id}")
                break

        return {
            "statusCode": 200,
            "body": json.dumps({
                "input_sku": input_sku,
                "recommended_product_id": best_product_id,
                "recommended_product_name": best_name,
                "score": best_score
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
