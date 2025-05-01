import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_data():
    engine = create_engine(os.getenv('DATABASE_URL'))

    query = """
    SELECT 
        od.order_id,
        od.product_id,
        od.unit_price,
        od.quantity,
        od.discount,
        o.customer_id,
        o.order_date,
        c.company_name
    FROM order_details od
    JOIN orders o ON od.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    """

    df = pd.read_sql(query, engine)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    return df

def create_features(df):
    df['order_total'] = df['unit_price'] * df['quantity'] * (1 - df['discount'])

    order_features = df.groupby(['order_id', 'customer_id']).agg({
        'order_total': 'sum',
        'discount': 'mean',
        'quantity': 'sum'
    }).reset_index()

    customer_features = df.groupby('customer_id').agg({
        'order_id': 'nunique',
        'order_total': ['mean', 'sum'],
        'discount': 'mean'
    }).reset_index()

    customer_features.columns = ['customer_id', 'total_orders', 'avg_order_value', 'total_spent', 'avg_discount']

    features = pd.merge(order_features, customer_features, on='customer_id')

    return features

def create_target_variable(df):
    discount_threshold = 0.15
    spending_threshold = df['order_total'].quantile(0.25)

    df['is_risky'] = ((df['discount'] > discount_threshold) & 
                      (df['order_total'] < spending_threshold)).astype(int)

    return df

def train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    class_weights = {0: 1, 1: 5}

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    return model, history

def explain_model(model, X_train, feature_names):
    explainer = shap.KernelExplainer(model.predict, X_train[:100])
    shap_values = explainer.shap_values(X_train[:100])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names)
    plt.savefig('shap_summary.png')
    plt.close()

    return shap_values

def main():
    print("Loading data...")
    df = load_data()

    print("Creating features...")
    features = create_features(df)
    features = create_target_variable(features)

    X = features[['order_total', 'discount', 'quantity', 'total_orders', 
                  'avg_order_value', 'total_spent', 'avg_discount']]
    y = features['is_risky']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training model...")
    model, history = train_model(X_train_scaled, y_train, X_test_scaled, y_test)

    print("\nModel Evaluation:")
    test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    print("\nGenerating model explanations...")
    shap_values = explain_model(model, X_train_scaled, X.columns)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # -------------------------
    # Riskli Siparişleri Kaydet (Sadece Riskli Olanlar)
    # -------------------------
    print("\nPredicting risky orders...")

    y_pred = model.predict(X_test_scaled)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()

    X_test_with_ids = X_test.copy()
    X_test_with_ids['order_id'] = features.loc[X_test.index, 'order_id'].values
    X_test_with_ids['customer_id'] = features.loc[X_test.index, 'customer_id'].values
    X_test_with_ids['predicted_is_risky'] = y_pred_labels

    risky_orders = X_test_with_ids[X_test_with_ids['predicted_is_risky'] == 1]

    print("\nRiskli Siparişler:")
    print(risky_orders)

    risky_orders.to_csv("risky_orders.csv", index=False)
    print("\nRiskli siparişler risky_orders.csv dosyasına kaydedildi.")

    # -------------------------
    # TÜM SİPARİŞLERİ Kaydet (Risksiz + Riskli)
    # -------------------------
    print("\nTüm siparişleri tahmin ediyoruz...")

    X_test_with_ids_all = X_test.copy()
    X_test_with_ids_all['order_id'] = features.loc[X_test.index, 'order_id'].values
    X_test_with_ids_all['customer_id'] = features.loc[X_test.index, 'customer_id'].values
    X_test_with_ids_all['predicted_is_risky'] = y_pred_labels

    X_test_with_ids_all.to_csv("all_orders_with_risk.csv", index=False)
    print("\nTüm siparişler all_orders_with_risk.csv dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
