import pytest
import json
import io
import tempfile
from fmnist_flask import app 

@pytest.fixture
def client():
    """Fixture to create a test client."""
    app.config['TESTING'] = True  # Enable testing mode
    client = app.test_client()
    return client

   
def test_predict_endpoint(client):
    """Test if the /predict endpoint returns a valid class label."""
    
    # Create a fake grayscale image (28x28 pixels)
    fake_image = io.BytesIO()
    import numpy as np
    import cv2

    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)  # Random grayscale image
    _, buffer = cv2.imencode(".png", image)
    fake_image.write(buffer.tobytes())
    fake_image.seek(0)  # Reset pointer to start

    # Write the image to a temporary file because OpenCV needs a real file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(fake_image.read())
        temp.seek(0)
        temp_name = temp.name

    with open(temp_name, "rb") as img:
        response = client.post(
            "/predict",
            content_type="multipart/form-data",
            data={"file": img},
        )

    assert response.status_code == 200  # Ensure the request was successful

    data = json.loads(response.data)
    assert "predicted_class" in data
    assert isinstance(data["predicted_class"], str)
