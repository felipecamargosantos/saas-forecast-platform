from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

def test_auth_workflow():
    test_user = "felipe.camargo@forecastlab.io"
    test_pass = "FelipeDev2026@Pass"
    client.post("/auth/register", json={"email": test_user, "password": test_pass})
    login_res = client.post("/auth/token", data={"username": test_user, "password": test_pass})
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

def test_unauthenticated_request():
    res = client.post("/forecast/upload-and-train")
    assert res.status_code in [401, 422]
