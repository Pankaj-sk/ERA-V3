import torch
import glob
from src.model import MNISTModel
from src.utils import evaluate_model

def test_model_accuracy():
    model = MNISTModel()
    # Find the most recent model file
    model_files = glob.glob('model_mnist_latest.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    
    # Load the model
    model.load_state_dict(torch.load(model_files[0]))
    
    # Test accuracy
    # Add your accuracy testing code here