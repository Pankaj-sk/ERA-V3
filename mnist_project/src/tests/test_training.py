import torch
import glob
from src.model import MNISTModel
from src.utils import evaluate_model

def test_model_accuracy():
    model = MNISTModel()
    # Find the most recent model file
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)  # Gets the most recent file
    
    model.load_state_dict(torch.load(latest_model))
    accuracy = evaluate_model(model)
    assert accuracy >= 0.8, f"Model accuracy {accuracy} is below 0.8" 