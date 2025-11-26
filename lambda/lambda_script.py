import boto3
import os
import pandas as pd
import json

# Clients
runtime = boto3.client('sagemaker-runtime')
bedrock = boto3.client('bedrock-runtime')  # AWS Bedrock

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "online-retail-xgb-serverless")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "meta.llama3-8b-instruct-v1:0")  # Example

# Fake product names mapping
PRODUCT_NAMES = {
    "71053": "Eco-friendly Water Bottle",
    "22752": "Wireless Earbuds",
    "21730": "Smart Fitness Tracker",
    "22633": "Noise-Cancelling Headphones",
    "22632": "Portable Charger",
    "84879": "Travel Backpack",
    "22745": "Bluetooth Speaker",
    "22748": "Desk Lamp",
    "22749": "Ergonomic Chair",
    "22310": "Stainless Steel Cookware Set",
    "84969": "Yoga Mat",
    "22623": "Electric Toothbrush",
    "22622": "Smart Thermostat",
    "21754": "Running Shoes",
    "21755": "Coffee Maker",
    "21777": "4K Monitor"
}

def lambda_handler(event, context):
    try:
        # Extract payload
        if "body" in event:
            payload = json.loads(event["body"])
        else:
            payload = event

        sku = payload.get("sku")
        if not sku:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'sku' in request"})}

        # --- SageMaker: get next-best-product ---
        df = pd.DataFrame([{
            "customer_id": 12345,
            "product_id": sku,
            "recency": 10,
            "frequency": 5,
            "monetary": 200.0,
            "unique_products": 3,
            "total_quantity": 7,
            "avg_order_value": 45.0,
            "product_popularity": 100,
            "product_avg_price": 50.0
        }])
        csv_payload = df.to_csv(index=False, header=False)

        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )
        result = sm_response['Body'].read().decode('utf-8')
        recommended_product_id = result.strip()  # assuming endpoint returns a single product_id
        recommended_product_name = PRODUCT_NAMES.get(recommended_product_id, "Recommended Product")

        # --- Bedrock: generate marketing message in Spanish ---
        prompt = f"""
        Eres un asistente de marketing. Un cliente está interesado en el producto SKU {sku}.
        Recomienda el siguiente mejor producto ({recommended_product_name}) en un tono amigable y persuasivo,
        animando al cliente a considerarlo para comprarlo. El mensaje debe estar en ESPAÑOL,
        pero el nombre del producto debe permanecer en inglés.
        """
        bedrock_response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType='application/json',
            body=json.dumps({"inputText": prompt})
        )
        marketing_text = json.loads(bedrock_response['body'].read().decode('utf-8'))['outputText']

        return {
            "statusCode": 200,
            "body": json.dumps({
                "input_sku": sku,
                "recommended_product_id": recommended_product_id,
                "recommended_product_name": recommended_product_name,
                "marketing_message": marketing_text
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
