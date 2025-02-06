import pytest
from fmnist_flask import app  

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to FMNIST Classifier" in response.data  # Modify based on your Flask app

def test_predict(client):
    with open("test_image.jpg", "rb") as img:
        response = client.post("/predict", data={"file": img}, content_type="multipart/form-data")
    
    assert response.status_code == 200
    assert "class" in response.json  # Modify based on your JSON response format
