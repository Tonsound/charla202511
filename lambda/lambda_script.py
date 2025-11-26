import boto3
import os
import pandas as pd
import json
import random

runtime = boto3.client('sagemaker-runtime')
bedrock = boto3.client('bedrock-runtime')

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "online-retail-xgb-serverless")
BEDROCK_MODEL_ID = "openai.gpt-oss-20b-1:0"

# Product names (unchanged)
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

candidate_products = list(product_names.keys())

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
        # Read SKU input
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

        # ----------------------------------
        # 🟩 SAGEMAKER PREDICTION (UNCHANGED)
        # ----------------------------------
        results = []
        for product_id in candidate_products:
            payload = generate_dummy_features(product_id)
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

        results.sort(key=lambda x: x[1], reverse=True)

        # Best alternative product
        for best_product_id, best_score in results:
            if best_product_id != input_sku:
                best_name = product_names.get(best_product_id, f"Product {best_product_id}")
                break

        # ---------------------------------------------------
        # 🟦 BEDROCK GPT-OSS-20B-1:0  (CHAT COMPLETION FORMAT)
        # ---------------------------------------------------
        prompt_message = f"""
        Eres un asistente de marketing. Un cliente está comprando el producto con SKU {input_sku}. 
        Recomienda el producto '{best_name}' como el siguiente mejor artículo para complementar su compra. 

        El mensaje debe:
        - Ser persuasivo, natural y amigable.
        - Estar completamente en ESPAÑOL.
        - Mantener el nombre del producto exactamente en inglés.
        - Ser breve (2–3 líneas máximo).
        - Incluir únicamente la recomendación, sin explicaciones adicionales ni texto extra.

        Entrega SOLO la recomendación.
        """

        native_request = {
            "messages": [
                {"role": "system",
                 "content": "Eres un experto en marketing que escribe mensajes persuasivos y cortos."},
                {"role": "user", "content": prompt_message}
            ],
            "max_completion_tokens": 120,
            "temperature": 0.7,
            "top_p": 0.9
        }

        br_response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(native_request)
        )

        br_body = json.loads(br_response["body"].read().decode("utf-8"))

        marketing_message = br_body["choices"][0]["message"]["content"].strip()

        # Final response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "input_sku": input_sku,
                "recommended_product_id": best_product_id,
                "recommended_product_name": best_name,
                "score": best_score,
                "marketing_message": marketing_message
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
