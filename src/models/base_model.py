import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self, model_name: str):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.debug = True

    def save_model(self, path: str):
        """Save model to a file."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from a file."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Model loaded from {path}")

    def debug_print(self, text: str):
        if self.debug:
            print(text)

