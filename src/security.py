import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError

SECRET_KEY = "felipe_forecast_platform_secret_key_2026_jwt_token"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

def gerar_hash_senha(senha_plana: str) -> str:
    senha_bytes = senha_plana.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(senha_bytes, salt).decode("utf-8")

def verificar_senha(senha_plana: str, senha_hasheada: str) -> bool:
    senha_bytes = senha_plana.encode("utf-8")[:72]
    hash_bytes = senha_hasheada.encode("utf-8")
    return bcrypt.checkpw(senha_bytes, hash_bytes)

def criar_token_jwt(dados: dict, minutos_expiracao: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    payload = dados.copy()
    expira_em = datetime.now(timezone.utc) + timedelta(minutes=minutos_expiracao)
    payload.update({"exp": expira_em})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decodificar_token_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return {}
