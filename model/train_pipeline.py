"""
Script para entrenar XGBoost en SageMaker usando un archivo de transacciones en formato Excel (XLSX).
Adaptado para el ejemplo de datos que enviaste (columnas: InvoiceNo, StockCode, Description,
Quantity, InvoiceDate, UnitPrice, CustomerID, Country).

Principales cambios respecto al script anterior:
- Lee un .xlsx en lugar de CSV.
- Normaliza UnitPrice reemplazando comas decimales por punto y convirtiendo a float.
- Parsea InvoiceDate intentando varios formatos (intenta dayfirst=True por defecto y cae
  a inferencias automáticas si falla).
- Usa StockCode como product_id y CustomerID como customer_id.
- Mantiene la lógica de feature engineering (por cliente-producto) y el pipeline de SageMaker.

Cómo usar:
- Ajusta S3_BUCKET y S3_PREFIX en la sección CONFIG.
- Coloca tu archivo Excel en ./data/transactions.xlsx o cambia la ruta.
- Ejecuta en un entorno con permisos SageMaker (SageMaker Studio / Notebook o definiendo SAGEMAKER_ROLE).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3
import sagemaker
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sklearn.model_selection import train_test_split
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.inputs import TrainingInput


# ----------------------- CONFIG -----------------------
REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = "tdr-artifacts"
S3_KEY = "tmp_files/charlaPUC/onlineRetail.xlsx"

LOCAL_DATA_DIR = "./data"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

LOCAL_FILE = os.path.join(LOCAL_DATA_DIR, "onlineRetail.xlsx")

# 🔥 Add this line
TRANSACTIONS_XLSX = LOCAL_FILE

s3 = boto3.client("s3", region_name=REGION)

print("Downloading file...")
s3.download_file(S3_BUCKET, S3_KEY, LOCAL_FILE)
print("File downloaded to:", LOCAL_FILE)


S3_PREFIX = "next-best-product"

# ===== XGBoost Training Configuration (Ranking) =====
INSTANCE_TYPE = "ml.m5.xlarge"
INSTANCE_COUNT = 1
MAX_RUN = 7200
XGBOOST_FRAMEWORK_VERSION = "1.7-1"

HYPERPARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "num_round": 200,
    "eta": 0.2,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

def prepare_and_upload_s3(sagemaker_session, df_features, bucket, prefix):
    """
    Prepare train.csv and validation.csv from final df_features and upload to S3.
    """

    # Remove ID columns only from training input (NOT from df – IDs needed later)
    cols = df_features.columns.tolist()
    feature_cols = [c for c in cols if c not in ["customer_id", "product_id", "label"]]

    # Prepare final CSV structure
    full_df = df_features[["customer_id", "product_id"] + feature_cols + ["label"]]

    # Split
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    # Local save
    local_train = "train.csv"
    local_val = "validation.csv"

    train_df.to_csv(local_train, index=False, header=True)
    val_df.to_csv(local_val, index=False, header=True)

    # Upload
    train_s3_path = f"s3://{bucket}/{prefix}/train/train.csv"
    val_s3_path = f"s3://{bucket}/{prefix}/validation/validation.csv"

    sagemaker_session.upload_data(local_train, bucket=bucket, key_prefix=f"{prefix}/train")
    sagemaker_session.upload_data(local_val, bucket=bucket, key_prefix=f"{prefix}/validation")

    print("Uploaded:")
    print(train_s3_path)
    print(val_s3_path)

    return train_s3_path, val_s3_path


def ensure_transactions_exists():
    """Si no existe el XLSX de transacciones, creamos un dataset sintético de ejemplo similar al formato provisto."""
    if os.path.exists(TRANSACTIONS_XLSX):
        print(f"Found transactions XLSX at {TRANSACTIONS_XLSX}")
        return

    print("No se encontró transactions.xlsx — generando dataset sintético de ejemplo...")
    np.random.seed(42)
    n_customers = 2000
    n_products = 200
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2025, 10, 31)
    days = (end_date - start_date).days

    rows = []
    for _ in range(20000):
        cust = np.random.randint(10000, 10000 + n_customers)
        prod = f"P{np.random.randint(1, n_products + 1):05d}"
        d = start_date + timedelta(days=np.random.randint(0, days + 1))
        qty = np.random.randint(1, 5)
        price = round(np.random.uniform(1, 200), 2)
        invoice = np.random.randint(500000, 600000)
        rows.append(
            [invoice, prod, f"Product {prod}", qty, d.strftime('%Y-%m-%d %H:%M'), price, cust, 'United Kingdom'])

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice",
                                     "CustomerID", "Country"])
    df.to_excel(TRANSACTIONS_XLSX, index=False)
    print(f"Synthetic transactions written to {TRANSACTIONS_XLSX}")

def load_transactions_from_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # <-- do NOT force dtype=str
    df.columns = [c.strip() for c in df.columns]

    # Normalize columns
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if 'invoice' in lc and 'date' in lc:
            colmap[c] = 'InvoiceDate'
        elif 'unit' in lc and 'price' in lc:
            colmap[c] = 'UnitPrice'
        elif 'quantity' in lc:
            colmap[c] = 'Quantity'
        elif 'customer' in lc:
            colmap[c] = 'CustomerID'
        elif 'stock' in lc:
            colmap[c] = 'StockCode'
        elif 'description' in lc:
            colmap[c] = 'Description'
        elif 'invoice' in lc and 'no' in lc:
            colmap[c] = 'InvoiceNo'
        elif 'country' in lc:
            colmap[c] = 'Country'

    df = df.rename(columns=colmap)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", 0), errors="coerce").fillna(0.0)
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0).astype(int)

    df = df.rename(
        columns={
            "CustomerID": "customer_id",
            "StockCode": "product_id",
            "InvoiceDate": "date",
            "UnitPrice": "price",
            "Quantity": "quantity",
        }
    )

    return df[["customer_id", "product_id", "date", "quantity", "price"]]


def build_features(df, holdout_days=30, negative_ratio=3, seed=42):
    """
    Build CUSTOMER × PRODUCT features without cartesian explosion.

    - Only positive pairs + sampled negatives
    - All indexes reset before merging (fixes ambiguity)
    - Memory-safe even for large datasets
    """

    df = df.copy()
    np.random.seed(seed)

    # -----------------------------
    # dtype cleanup
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["revenue"] = df["quantity"] * df["price"]

    df = df.dropna(subset=["customer_id", "product_id", "date"])

    # -----------------------------
    # Train vs Holdout
    # -----------------------------
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    df_train = df[df["date"] <= cutoff_date]
    df_holdout = df[df["date"] > cutoff_date]

    # -----------------------------
    # CUSTOMER FEATURES
    # -----------------------------
    g = df_train.groupby("customer_id")

    customer_features = g.agg(
        last_purchase=("date", "max"),
        frequency=("date", "count"),
        monetary=("revenue", "sum"),
        unique_products=("product_id", "nunique"),
        total_quantity=("quantity", "sum"),
    ).reset_index()  # FIX: ensure customer_id is a column

    customer_features["recency"] = (
        cutoff_date - customer_features["last_purchase"]
    ).dt.days

    customer_features["avg_order_value"] = (
        customer_features["monetary"] /
        customer_features["frequency"].replace(0, 1)
    )

    customer_features = customer_features.fillna(0)

    # -----------------------------
    # PRODUCT FEATURES
    # -----------------------------
    product_features = (
        df_train.groupby("product_id")
        .agg(
            product_popularity=("quantity", "sum"),
            product_avg_price=("price", "mean"),
        )
        .reset_index()  # FIX: ensure product_id is a column
        .fillna(0)
    )

    # -----------------------------
    # POSITIVE PAIRS
    # -----------------------------
    pos_pairs = (
        df_train.groupby(["customer_id", "product_id"])
        .size()
        .reset_index()[["customer_id", "product_id"]]
    )
    pos_pairs["label"] = 1

    # -----------------------------
    # NEGATIVE SAMPLING
    # -----------------------------
    all_customers = df_train["customer_id"].unique()
    all_products = df_train["product_id"].unique()

    existing_pairs = set(zip(pos_pairs.customer_id, pos_pairs.product_id))

    neg_samples = []
    target_neg = len(pos_pairs) * negative_ratio

    while len(neg_samples) < target_neg:
        c = np.random.choice(all_customers)
        p = np.random.choice(all_products)
        if (c, p) not in existing_pairs:
            neg_samples.append((c, p))
            existing_pairs.add((c, p))

    neg_pairs = pd.DataFrame(neg_samples, columns=["customer_id", "product_id"])
    neg_pairs["label"] = 0

    # -----------------------------
    # Combine pairs
    # -----------------------------
    df_pairs = pd.concat([pos_pairs, neg_pairs], ignore_index=True)

    # -----------------------------
    # Merge features (all indexes reset)
    # -----------------------------
    df_features = (
        df_pairs
        .merge(customer_features, on="customer_id", how="left")
        .merge(product_features, on="product_id", how="left")
        .fillna(0)
    )

    # -----------------------------
    # HOLDOUT label (for validation)
    # -----------------------------
    holdout_labels = (
        df_holdout.groupby(["customer_id", "product_id"])
        .size()
        .reset_index(name="bought")
    )
    holdout_labels["holdout_label"] = 1
    holdout_labels = holdout_labels[["customer_id", "product_id", "holdout_label"]]

    # FIX: ensure no index levels cause ambiguity
    holdout_labels = holdout_labels.reset_index(drop=True)

    df_features = df_features.merge(
        holdout_labels, on=["customer_id", "product_id"], how="left"
    )
    df_features["holdout_label"] = df_features["holdout_label"].fillna(0).astype(int)

    print("Final shape:", df_features.shape)
    return df_features


def train_xgboost_sagemaker(
    sagemaker_session,
    role,
    train_s3_uri,
    val_s3_uri,
    bucket,
    prefix
):
    from sagemaker.inputs import TrainingInput
    from sagemaker.estimator import Estimator
    from sagemaker import image_uris

    # Built-in XGBoost container image
    container = image_uris.retrieve(
        framework="xgboost",
        version=XGBOOST_FRAMEWORK_VERSION,
        region=sagemaker_session.boto_region_name,
    )

    xgb = Estimator(
        image_uri=container,
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        hyperparameters=HYPERPARAMS,
        max_run=MAX_RUN,
        output_path=f"s3://{bucket}/{prefix}/output",
        sagemaker_session=sagemaker_session,
    )

    train_input = TrainingInput(train_s3_uri, content_type="text/csv")
    val_input   = TrainingInput(val_s3_uri, content_type="text/csv")

    xgb.fit({"train": train_input, "validation": val_input}, wait=True)

    print("✔ Training completed")
    return xgb


TRAIN_SCRIPT_CONTENT = r"""
import os
import pandas as pd
import xgboost as xgb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--num_round', type=int, default=200)
args, _ = parser.parse_known_args()


train_path = '/opt/ml/input/data/train/train.csv'
val_path = '/opt/ml/input/data/validation/validation.csv'


train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)


label_col = 'label'
if label_col not in train_df.columns:
raise ValueError('El CSV de entrenamiento debe contener la columna `label` con 0/1')


X_train = train_df.drop(columns=['customer_id','product_id', label_col], errors='ignore')
y_train = train_df[label_col]
X_val = val_df.drop(columns=['customer_id','product_id', label_col], errors='ignore')
if 'label' in val_df.columns:
y_val = val_df[label_col]
else:
y_val = None


# Convertir a DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)


dval = xgb.DMatrix(X_val, label=y_val) if y_val is not None else None


params = {
'objective': 'binary:logistic',
'eval_metric': 'auc',
}


watchlist = [(dtrain, 'train')]
if dval is not None:
watchlist.append((dval, 'validation'))


model = xgb.train(params, dtrain, num_boost_round=args.num_round, evals=watchlist)


model_path = '/opt/ml/model/xgboost-model'
model.save_model(model_path)
print('Model saved to', model_path)
"""

ensure_transactions_exists()
transactions = load_transactions_from_xlsx(TRANSACTIONS_XLSX)
transactions.head()

print('Building features...')
df_features = build_features(transactions, holdout_days=30)
print('Features shape:', df_features.shape)


sagemaker_session = sagemaker.Session()
try:
    role = sagemaker.get_execution_role()
except Exception:
    role = os.environ.get('SAGEMAKER_ROLE')
if role is None:
    raise RuntimeError("No se pudo obtener el role de SageMaker. Ejecuta esto en un entorno SageMaker o define SAGEMAKER_ROLE en env.")


train_s3_uri, val_s3_uri = prepare_and_upload_s3(
    sagemaker_session=sagemaker_session,
    df_features=df_features,
    bucket=S3_BUCKET,
    prefix=S3_PREFIX
)

xgboost_model = train_xgboost_sagemaker(
    sagemaker_session=sagemaker_session,
    role=role,
    train_s3_uri=train_s3_uri,
    val_s3_uri=val_s3_uri,
    bucket=S3_BUCKET,
    prefix=S3_PREFIX
)

sm = boto3.client("sagemaker")

endpoint_name = "online-retail-xgb-serverless"

# 1. Delete endpoint if exists
try:
    sm.delete_endpoint(EndpointName=endpoint_name)
    print("Deleted endpoint:", endpoint_name)
except sm.exceptions.ClientError as e:
    if "Could not find endpoint" in str(e):
        print("Endpoint does not exist, OK")
    else:
        raise e

# 2. Delete endpoint config
try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("Deleted endpoint config:", endpoint_name)
except sm.exceptions.ClientError as e:
    if "Could not find endpoint configuration" in str(e):
        print("Endpoint config does not exist, OK")
    else:
        raise e

model_artifact = xgboost_model.model_data
print("Model artifact:", model_artifact)

# Force SageMaker to use your bucket
serverless_model = XGBoostModel(
    model_data=model_artifact,
    role=role,
    framework_version="1.7-1",
    sagemaker_session=sagemaker_session
)

serverless_model.bucket = "tdr-artifacts"  # ← FIX

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5
)

predictor = serverless_model.deploy(
    endpoint_name="online-retail-xgb-serverless",
    serverless_inference_config=serverless_config
)

print("✔ Serverless endpoint deployed:", predictor.endpoint_name)
