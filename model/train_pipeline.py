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

# ----------------------- CONFIG -----------------------
REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = "tdr-artifacts"
S3_KEY = "tmp_files/charlaPUC/onlineRetail.xlsx"

LOCAL_DATA_DIR = "../data"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

LOCAL_FILE = os.path.join(LOCAL_DATA_DIR, "onlineRetail.xlsx")

# 🔥 Add this line
TRANSACTIONS_XLSX = LOCAL_FILE

s3 = boto3.client("s3", region_name=REGION)

print("Downloading file...")
s3.download_file(S3_BUCKET, S3_KEY, LOCAL_FILE)
print("File downloaded to:", LOCAL_FILE)


INSTANCE_TYPE = "ml.m3.medium"
INSTANCE_COUNT = 1
MAX_RUN = 3600 * 2
XGBOOST_FRAMEWORK_VERSION = "1.5-1"
HYPERPARAMS = {
"objective": "binary:logistic",
"num_round": 200,
"eta": 0.1,
"max_depth": 6,
"subsample": 0.8,
"verbosity": 1,
}

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


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


def build_features(df, holdout_days=30):
    """
    Build CUSTOMER × PRODUCT features for next-best-product prediction.
    Output: each row is (customer_id, product_id) with RFM + product stats + label.
    """

    df = df.copy()

    # Proper dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["revenue"] = df["quantity"] * df["price"]

    # Drop invalid rows
    df = df.dropna(subset=["customer_id", "product_id", "date"])

    # -----------------------------
    # Train / Holdout
    # -----------------------------
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    df_train = df[df["date"] <= cutoff_date]
    df_holdout = df[df["date"] > cutoff_date]

    # -----------------------------
    # CUSTOMER RFM FEATURES
    # -----------------------------
    customer_last = df_train.groupby("customer_id")["date"].max().rename("last_purchase")
    customer_freq = df_train.groupby("customer_id")["date"].count().rename("frequency")
    customer_monetary = df_train.groupby("customer_id")["revenue"].sum().rename("monetary")
    customer_recency = (cutoff_date - customer_last).dt.days.rename("recency")
    customer_unique_products = (
        df_train.groupby("customer_id")["product_id"].nunique().rename("unique_products")
    )
    customer_total_qty = df_train.groupby("customer_id")["quantity"].sum().rename("total_quantity")

    customer_features = pd.concat(
        [
            customer_last,
            customer_recency,
            customer_freq,
            customer_monetary,
            customer_unique_products,
            customer_total_qty,
        ],
        axis=1,
    ).fillna(0)
    customer_features["avg_order_value"] = (
        customer_features["monetary"] / customer_features["frequency"].replace(0, 1)
    )
    customer_features = customer_features.reset_index()

    # -----------------------------
    # PRODUCT FEATURES
    # -----------------------------
    product_popularity = (
        df_train.groupby("product_id")["quantity"]
        .sum()
        .rename("product_popularity")
        .reset_index()
    )

    product_avg_price = (
        df_train.groupby("product_id")["price"]
        .mean()
        .rename("product_avg_price")
        .reset_index()
    )

    # -----------------------------
    # CUSTOMER × PRODUCT CANDIDATES
    # -----------------------------
    all_customers = df_train["customer_id"].unique()
    all_products = df_train["product_id"].unique()

    df_pairs = (
        pd.MultiIndex.from_product([all_customers, all_products], names=["customer_id", "product_id"])
        .to_frame(index=False)
    )

    # Merge customer and product features
    df_features = df_pairs.merge(customer_features, on="customer_id", how="left")
    df_features = df_features.merge(product_popularity, on="product_id", how="left")
    df_features = df_features.merge(product_avg_price, on="product_id", how="left")
    df_features = df_features.fillna(0)

    # -----------------------------
    # LABEL: Did customer buy product in the next period?
    # -----------------------------
    purchases_next = (
        df_holdout.groupby(["customer_id", "product_id"])
        .size()
        .reset_index(name="bought")
    )
    purchases_next["label"] = 1
    purchases_next = purchases_next[["customer_id", "product_id", "label"]]

    df_features = df_features.merge(purchases_next, on=["customer_id", "product_id"], how="left")
    df_features["label"] = df_features["label"].fillna(0).astype(int)

    print("Final feature matrix shape:", df_features.shape)
    return df_features

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

def train_xgboost_sagemaker(sagemaker_session, role, train_s3_uri, val_s3_uri, bucket, prefix):
    print("▶ Starting XGBoost (Next Best Product) training job...")

    container = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=sagemaker_session.boto_region_name,
        version="1.7-1"
    )

    xgb = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{bucket}/{prefix}/model_output",
        sagemaker_session=sagemaker_session
    )

    # CLASSIFICATION
    xgb.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=200,
        eta=0.2,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8
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
print('model saved to', model_path)
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