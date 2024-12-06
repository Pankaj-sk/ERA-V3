import os
import torch
from src.model import MNISTModel
from src.utils import evaluate_model

def test_model_accuracy():
    # Get the src directory path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, 'model_mnist_latest.pth')
    
    print(f"Looking for model at: {model_path}")  # Debug print
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")
    
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    accuracy = evaluate_model(model)
    assert accuracy >= 0.8, f"Model accuracy {accuracy} is below 0.8"