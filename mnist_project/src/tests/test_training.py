import torch
from src.model import MNISTModel
from src.utils import evaluate_model

def test_model_accuracy():
    model = MNISTModel()
    model.load_state_dict(torch.load('model_mnist_latest.pth'))
    accuracy = evaluate_model(model)
    assert accuracy >= 0.8, f"Model accuracy {accuracy} is below 0.8" 