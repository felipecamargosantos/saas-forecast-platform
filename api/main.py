import uuid
from typing import Dict, Any
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database import get_db, User
from src.security import gerar_hash_senha, verificar_senha, criar_token_jwt, decodificar_token_jwt
from src.schemas import UserCreate, TokenResponse, JobStatusResponse, ForecastRequest, ForecastResponse
from src.worker import run_lstm_training

app = FastAPI(title="SaaS Forecast Platform", version="1.0.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
jobs_db: Dict[str, Dict[str, Any]] = {}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decodificar_token_jwt(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Token invalido")
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Usuario inativo")
    return user

@app.get("/health")
def health():
    return {"status": "HEALTHY", "service": "saas-forecast-platform"}

@app.post("/auth/register", status_code=201)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email ja cadastrado")
    user = User(email=user_data.email, hashed_password=gerar_hash_senha(user_data.password))
    db.add(user)
    db.commit()
    return {"message": "Usuario registrado"}

@app.post("/auth/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verificar_senha(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Credenciais invalidas")
    token = criar_token_jwt({"sub": user.email, "user_id": user.id})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/forecast/upload-and-train", status_code=status.HTTP_202_ACCEPTED, response_model=JobStatusResponse)
async def upload_and_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Somente CSV permitido")
    contents = await file.read()
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "user_id": current_user.id,
        "status": "QUEUED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": "Enfileirado",
        "metrics": None
    }
    background_tasks.add_task(run_lstm_training, job_id, contents, jobs_db)
    return JobStatusResponse(
        job_id=job_id,
        status="QUEUED",
        created_at=jobs_db[job_id]["created_at"],
        message="Upload recebido"
    )

@app.get("/forecast/jobs/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str, current_user: User = Depends(get_current_user)):
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job nao encontrado")
    if job["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Acesso negado")
    return JobStatusResponse(
        job_id=job_id, status=job["status"], created_at=job["created_at"],
        metrics=job.get("metrics"), message=job["message"]
    )

@app.post("/forecast/predict/{job_id}", response_model=ForecastResponse)
def predict(job_id: str, payload: ForecastRequest, current_user: User = Depends(get_current_user)):
    job = jobs_db.get(job_id)
    if not job or job["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Modelo nao concluido")
    model = tf.keras.models.load_model(job["model_path"])
    current_window = np.array(job["last_window"]).reshape(1, 14, 1)
    preds = []
    for _ in range(payload.horizon_days):
        step = model(current_window, training=False).numpy()[0][0]
        preds.append(step)
        current_window = np.append(current_window[:, 1:, :], [[[step]]], axis=1)
    s_min, s_max = job["scaler_min"], job["scaler_max"]
    unscaled = [round(float(p * (s_max - s_min) + s_min), 2) for p in preds]
    return ForecastResponse(job_id=job_id, horizon_days=payload.horizon_days, forecast_values=unscaled)
