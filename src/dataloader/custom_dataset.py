# /src/data/custom_dataset.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class CustomDataset(ABC, Dataset):
    def __init__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    @abstractmethod
    def _process_csv(self):
        pass
