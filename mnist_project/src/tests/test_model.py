import torch
import pytest
from src.model import MNISTModel

def test_model_architecture():
    model = MNISTModel()
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"

def test_model_parameters():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has {total_params} parameters, should be < 25000" 