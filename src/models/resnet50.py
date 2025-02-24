import torch
import torch.nn as nn
import torchvision.models as models
from src.models.base_model import BaseModel

class ResNet50(BaseModel):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResNet50, self).__init__(model_name="ResNet50 Model")
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify final layer for custom classes
            
        # # Add Dropout Layer after Layer3
        # self.model.layer3 = nn.Sequential(
        #     self.model.layer3,
        #     nn.Dropout(p=0.5)  # Adding dropout
        # )
        
        # Modify Fully Connected Layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Adding dropout before final layer
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, y):
        return self.model(x), y
    
    def save_model(self, model_saved_path):
        super().save_model(model_saved_path)

# Example Usage
if __name__ == "__main__":
    model = ResNet50(num_classes=3, pretrained=True)  # Change num_classes as needed
    x = torch.randn(1, 3, 224, 224)  # Example input
    output, _ = model(x, x)
    print(output.shape)  # Expected output: torch.Size([1, 10])
    