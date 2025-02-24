# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluations.monte_carlo import MultiLabelEvaluator as Evaluator
from src.utils.filename_manager import FilenameManager
from src.utils.results_writer import MetricsTracker
from src.utils.losses import BNN_BCEWithLogitsLoss, BNN_CrossEntropyLoss
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.num_samples = len(self.dataloader.dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load configuration settings
        self.training_config = config['training']
        self.optimizer_config = self.training_config['optimizer']
        self.learning_rate = self.training_config['learning_rate']
        self.loss_function_config = self.training_config['loss_function']
        self.num_epochs = self.training_config['num_epochs']

        # Initialize optimizer and loss function
        self.optimizer = self._initialize_optimizer()
        self.loss_function = self._initialize_loss_function()

        self.evaluation_metrics = Evaluator()
        self.log_file_path = FilenameManager().get_filename('training_log')
        self.results_writer = MetricsTracker(self.log_file_path)

    def _initialize_optimizer(self):
        if self.optimizer_config == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config}")

    
    def _initialize_loss_function(self):
        if self.loss_function_config == "bnn_cross_entropy":
            return BNN_CrossEntropyLoss
        elif self.loss_function_config == "bnn_binary_cross_entropy":
            print("BNN_CrossEntropyLoss")
            return BNN_CrossEntropyLoss
        elif self.loss_function_config == "nn.CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_config}")

    def train(self, val_loader):

        for epoch in range(1, self.num_epochs+1):
            print(f"Epoch [{epoch}/{self.num_epochs}]")

            val_loss, val_metrics = self.validate_epoch(val_loader=val_loader)
            train_loss, train_metrics = self.train_epoch(epoch=epoch)

            model_saved_path = FilenameManager().generate_model_filename(epoch=epoch, learning_rate=self.learning_rate, extension='pth')            
            self.model.save_model(model_saved_path)

            self.results_writer.update(epoch=epoch, batch=None, train_loss=train_loss, val_loss=val_loss, train_metrics=train_metrics, val_metrics=val_metrics)
            self.results_writer.save()


    def train_epoch(self, epoch):
        self.evaluation_metrics.reset_metrics()
        self.model.train()

        running_loss = 0.0

        for batch, (images, _, y)  in enumerate(tqdm(self.dataloader, desc="Processing Training Batches")):
            images, y = images.to(self.device), y.to(self.device)
            if epoch == 1 and batch == 0:
                print(f'{images.shape}, {y.shape}')

            self.model.train()
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred, var = self.model(images, _)
            loss = self.loss_function(pred, y, var)
            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.evaluation_metrics.update_metrics(batch_y_true=y, batch_y_pred=pred, batch_y_variance=var)
        
        epoch_loss = running_loss / len(self.dataloader)
        print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics
        

    def validate_epoch(self, val_loader):
        self.model.eval()
        self.evaluation_metrics.reset_metrics()

        running_loss = 0.0
        with torch.no_grad():
            for batch, (images, _, y)  in enumerate(tqdm(val_loader, desc="Processing Validation Batches")):
                images, y = images.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {y.shape}')

                # Compute prediction and loss
                pred, var = self.model(images, _)
                loss = self.loss_function(pred, y, var)
                running_loss += loss.item()

                self.evaluation_metrics.update_metrics(batch_y_true=y, batch_y_pred=pred, batch_y_variance=var)

            
        epoch_loss = running_loss / len(val_loader)
        print(f"Validation Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        print(f"Validation:\n")
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics

