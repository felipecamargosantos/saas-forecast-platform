from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    email: str = Field(..., description="E-mail corporativo")
    password: str = Field(..., min_length=6, description="Senha de acesso")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    metrics: Optional[Dict[str, float]] = None
    message: str

class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=14, ge=1, le=60)

class ForecastResponse(BaseModel):
    job_id: str
    horizon_days: int
    forecast_values: List[float]
