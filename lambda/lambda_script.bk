import boto3
import os
import pandas as pd
import json

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "online-retail-xgb-serverless")

def lambda_handler(event, context):
    try:
        # Extract the JSON body (API Gateway proxy).
        if "body" in event:
            payload = json.loads(event["body"])
        else:
            payload = event  # For direct testing from console

        # Build CSV payload
        df = pd.DataFrame([payload])
        csv_payload = df.to_csv(index=False, header=False)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )

        result = response['Body'].read().decode('utf-8')
        prediction = json.loads(result) if result.strip().startswith('[') else result

        return {
            "statusCode": 200,
            "body": json.dumps({
                "input": payload,
                "prediction": prediction
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
