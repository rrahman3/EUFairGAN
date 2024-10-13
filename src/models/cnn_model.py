import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel

class CNNModel(BaseModel): #input shape (None, 3, Px, Py)
    def __init__(self, num_classes=3, drop_rate=0.50, hidden_layer=128):
        super(CNNModel, self).__init__(model_name="CNN Model")
        self.hidden_layer = hidden_layer
        self.num_classes = num_classes
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(p=drop_rate)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(p=drop_rate)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu_3 = nn.ReLU()
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        self.dropout_3 = nn.Dropout(p=drop_rate)
        self.flatten = nn.Flatten()
        self.dropout_4 = nn.Dropout(p=drop_rate)

        # self.dense_gender = nn.Linear(1, 32)
        # self.relu_4 = nn.ReLU()
        # self.dropout_5 = nn.Dropout(p=drop_rate)
        # 14*14*128
        self.dense_1 = nn.LazyLinear(self.hidden_layer)
        self.variance = nn.Linear(self.hidden_layer, 1)
        # self.softplus = nn.Softplus()
        self.dense_2 = nn.Linear(self.hidden_layer, self.num_classes)

    def forward(self, images, genders):
        # images, genders = inputs
        # print(f"Image_Shape {images.shape}")
        # print(f"Gender Shape {genders.shape}")

        x = self.conv_1(images)
        x = self.relu_1(x)
        x = self.max_pool_1(x)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.max_pool_2(x)
        x = self.dropout_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.max_pool_3(x)
        x = self.dropout_3(x)
        x = self.flatten(x)
        x = self.dropout_4(x)

        # y = self.dense_gender(genders)
        # y = self.relu_4(y)
        # y = self.dropout_5(y)

        # print(f"before concat {x.shape}, {y.shape}")
        # x = torch.cat([x, y], dim=-1)
        # print(f"after concat {x.shape}")
        x = self.dense_1(x)
        variance = self.variance(x)
        variance = F.softplus(variance)
        x = self.dense_2(x)
        return x, variance


    def save_model(self, model_saved_path):
        super().save_model(model_saved_path)


