import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models

def run_lstm_training(job_id: str, csv_bytes: bytes, jobs_db: dict):
    try:
        jobs_db[job_id]["status"] = "PROCESSING"
        jobs_db[job_id]["message"] = "Processando janelas e tensores"
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if "value" not in df.columns:
            raise ValueError("Coluna value ausente")
        valores = pd.to_numeric(df["value"], errors="coerce").dropna().values.reshape(-1, 1)
        if len(valores) < 40:
            raise ValueError("Dataset requer minimo de 40 observacoes")
        split = int(len(valores) * 0.8)
        tr_raw, te_raw = valores[:split], valores[split:]
        sc = MinMaxScaler(feature_range=(0, 1))
        tr_scaled = sc.fit_transform(tr_raw)
        te_scaled = sc.transform(te_raw)
        w = 14
        X_tr, y_tr = [], []
        for i in range(len(tr_scaled) - w):
            X_tr.append(tr_scaled[i : i + w])
            y_tr.append(tr_scaled[i + w])
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        m = models.Sequential([
            layers.Input(shape=(w, 1)),
            layers.LSTM(24, activation="tanh"),
            layers.Dense(12, activation="relu"),
            layers.Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_tr, y_tr, epochs=15, batch_size=16, verbose=0)
        f_scaled = np.vstack((tr_scaled[-w:], te_scaled))
        X_te, y_te = [], []
        for i in range(len(f_scaled) - w):
            X_te.append(f_scaled[i : i + w])
            y_te.append(f_scaled[i + w])
        X_te, y_te = np.array(X_te), np.array(y_te)
        p = sc.inverse_transform(m.predict(X_te, verbose=0))
        act = sc.inverse_transform(y_te)
        rmse = float(np.sqrt(mean_squared_error(act, p)))
        mae = float(mean_absolute_error(act, p))
        m_path = f"models_store/{job_id}.keras"
        m.save(m_path)
        jobs_db[job_id]["status"] = "COMPLETED"
        jobs_db[job_id]["message"] = "Treinamento finalizado"
        jobs_db[job_id]["metrics"] = {"rmse": round(rmse, 2), "mae": round(mae, 2)}
        jobs_db[job_id]["last_window"] = f_scaled[-w:].tolist()
        jobs_db[job_id]["scaler_min"] = float(sc.data_min_[0])
        jobs_db[job_id]["scaler_max"] = float(sc.data_max_[0])
        jobs_db[job_id]["model_path"] = m_path
    except Exception as e:
        jobs_db[job_id]["status"] = "FAILED"
        jobs_db[job_id]["message"] = str(e)
