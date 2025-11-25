import boto3
import os
import pandas as pd
import json

# SageMaker runtime client.
runtime = boto3.client('sagemaker-runtime')

# Endpoint name (already deployed)
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "online-retail-xgb-serverless")

def lambda_handler(event, context):
    try:
        # Example: event contains JSON like your Excel rows
        # {
        #   "customer_id": 16419,
        #   "product_id": 90159,
        #   "last_purchase": "2011-08-19 12:33:00",
        #   "frequency": 52,
        #   ...
        # }
        df = pd.DataFrame([event])

        # Convert to CSV without header/index (XGBoost expects this)
        csv_payload = df.to_csv(index=False, header=False)

        # Invoke SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )

        # Read prediction
        result = response['Body'].read().decode('utf-8')
        prediction = json.loads(result) if result.strip().startswith('[') else result

        return {
            "statusCode": 200,
            "body": json.dumps({
                "input": event,
                "prediction": prediction
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
