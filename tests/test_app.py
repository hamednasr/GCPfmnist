import pytest
import json
import io
from fmnist_flask import app 

@pytest.fixture
def client():
    """Fixture to create a test client."""
    app.config['TESTING'] = True  # Enable testing mode
    client = app.test_client()
    return client

   
def test_predict_endpoint(client):
    """Test if the /predict endpoint returns a valid class label."""
    
    # Create a fake image (28x28 grayscale)
    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    fake_image.name = "test.png"  # Assign a name to simulate a real file

    response = client.post(
        "/predict",
        content_type="multipart/form-data",
        data={"file": (fake_image, "test.png")},
    )

    assert response.status_code == 200  # Ensure the request was successful

    data = json.loads(response.data)
    assert "predicted_class" in data
    assert isinstance(data["predicted_class"], str)
    

def test_invalid_request(client):
    """Test the API with an invalid request (no file uploaded)."""
    response = client.post("/predict")
    
    assert response.status_code == 400  # Should return an error
    data = json.loads(response.data)
    assert "error" in data  # Error message should be present
