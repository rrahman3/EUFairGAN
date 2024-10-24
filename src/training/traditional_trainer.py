# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluations.traditional_evaluator import TraditionalEvaluator
from src.utils.losses import crossentropy_loss
from src.utils.filename_manager import FilenameManager
from src.utils.results_writer import MetricsTracker
from tqdm import tqdm

class TraditionalTrainer:
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

        self.evaluation_metrics = TraditionalEvaluator()
        self.log_file_path = FilenameManager().get_filename('training_log')
        self.results_writer = MetricsTracker(self.log_file_path)

    def _initialize_optimizer(self):
        if self.optimizer_config == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config}")

    def _initialize_loss_function(self):
        if self.loss_function_config == "cross_entropy":
            return crossentropy_loss
        elif self.loss_function_config == "nn.CrossEntropyLoss":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_config}")

    def train(self, val_loader):

        for epoch in range(1, self.num_epochs+1):
            print(f"Epoch [{epoch}/{self.num_epochs}]")

            train_loss, train_metrics = self.train_epoch(epoch=epoch)
            val_loss, val_metrics = self.validate_epoch(val_loader=val_loader)

            model_saved_path = FilenameManager().generate_model_filename(epoch=epoch, learning_rate=self.learning_rate, extension='pth')            
            self.model.save_model(model_saved_path)

            self.results_writer.update(epoch=epoch, batch=None, train_loss=train_loss, val_loss=val_loss, train_metrics=train_metrics, val_metrics=val_metrics)
            self.results_writer.save()



    def train_epoch(self, epoch):
        self.evaluation_metrics.reset_metrics()
        self.model.train()

        running_loss = 0.0
        batch = 0
        for images, genders, y  in tqdm(self.dataloader):
            images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
            if epoch == 1 and batch == 0:
                print(f'{images.shape}, {genders.shape}, {y.shape}')

            self.model.train()
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred = self.model(images)
            loss = self.loss_function(y, pred)
            running_loss += loss.item() * images.shape[0]
            print(pred)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.evaluation_metrics.update_metrics(y_true=y, y_pred=pred)

            batch += 1
        
        epoch_loss = running_loss / self.num_samples
        print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics
        

    def validate_epoch(self, val_loader):
        self.model.eval()
        self.evaluation_metrics.reset_metrics()

        running_loss = 0.0
        batch = 0
        with torch.no_grad():
            for images, genders, y  in tqdm(val_loader):
                images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {genders.shape}, {y.shape}')
                batch += 1

                # Compute prediction and loss
                pred = self.model(images)
                loss = self.loss_function(y, pred)
                running_loss += loss.item() * images.shape[0]

                self.evaluation_metrics.update_metrics(y_true=y, y_pred=pred)

            
        epoch_loss = running_loss / len(val_loader)
        print(f"Validation Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        print(f"Validation:\n")
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics

