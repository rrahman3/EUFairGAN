import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# sys.path.insert(0, os.path.abspath('../../'))
# print(sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assume that 'src' is in the parent directory of the parent directory of the current file.
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
print("Project root added to sys.path:", project_root)

from src.models.base_model import BaseModel

class UTKFaceAgeModel(BaseModel): #input shape (None, 3, Px, Py)
    def __init__(self, task='regression', drop_rate=0.50, hidden_layer=128):
        super(UTKFaceAgeModel, self).__init__(model_name="UTKFaceAgeModel")
        self.task = task
        self.hidden_layer = hidden_layer

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
        self.dense_1 = nn.LazyLinear(self.hidden_layer)
        
        if self.task == 'regression':
            # Regression: Predict age (scalar) and its log variance
            self.age_head = nn.Linear(self.hidden_layer, 1)
            self.log_var_head = nn.Linear(self.hidden_layer, 1)
        
        elif self.task == 'classification':
            # Classification: Predict class logits and a log variance for each class.
            # Assumes self.num_classes is defined in BaseModel.
            self.logit_head = nn.Linear(self.hidden_layer, self.num_classes)
            self.log_var_head = nn.Linear(self.hidden_layer, self.num_classes)
        else:
            raise ValueError("Task must be either 'regression' or 'classification'.")
        
        
    def forward(self, images):
        """
        Forward pass for both tasks.
        

        Args:
            images (Tensor): Input image batch of shape (batch, 3, H, W).
            genders (Tensor, optional): Optional gender input (if used); currently ignored.
            mc_samples (int): Number of Monte Carlo samples (only used in classification).
        
        Returns:
            For regression:
                (age, log_var) – both of shape (batch, 1)
            For classification:
                Tensor of shape (mc_samples, batch, num_classes) containing MC-sampled logits.
        """

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
        x = self.dense_1(x)
        
        if self.task == 'regression':
            # Heteroscedastic regression: predict age and log variance.
            age = self.age_head(x)
            log_var = self.log_var_head(x)
            # Optionally, you could use softplus on log_var to ensure numerical stability.
            # However, many loss functions expect the raw log variance.
            return age, log_var
        
        variance = self.variance(x)
        variance = F.softplus(variance)
        x = self.dense_2(x)
        return x, variance


    def save_model(self, model_saved_path):
        super().save_model(model_saved_path)


import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, hamming_loss
import numpy as np
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error

    
class MultiLabelEvaluator:
    def __init__(self, n_labels=14, threshold=0.5, eval_type="multilabel"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_labels = n_labels  # Number of labels per sample
        self.threshold = threshold  # Threshold for binary classification
        self.eval_type = eval_type.lower()
        self.reset_metrics()

    def reset_metrics(self):
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        self.y_au_score = torch.tensor([]).to(self.device)

    def update_metrics(self, batch_y_true, batch_y_pred, batch_y_variance): # y_ture is the binary form of the y_true (None, 1, num_classes) # [[0, 0, 1, 0]] (1, 1, 4)

        # Get predictions

        self.y_true = torch.cat((self.y_true, batch_y_true.clone().detach()), 0)
        self.y_score = torch.cat((self.y_score, batch_y_pred.clone().detach()), 0)

        self.y_au_score = torch.cat((self.y_au_score, batch_y_variance.clone().detach()), 0)


    def apply_threshold(self, y_pred):
        """
        Apply threshold to the predicted probabilities to convert to binary labels.
        """
        return (y_pred >= self.threshold).astype(int)

    def compute_epoch_metrics(self, epoch_y_pred=None, epoch_y_true=None, epoch_y_au_score=None):
        """
        Compute metrics for the epoch according to eval_type.
        
        Returns:
            A dictionary containing the computed metrics.
        """
        # Use provided epoch predictions if given; otherwise use stored ones.
        if epoch_y_pred is None:
            y_score = self.y_score.detach().cpu().numpy()
        else:
            y_score = epoch_y_pred
        if epoch_y_true is None:
            y_true = self.y_true.detach().cpu().numpy()
        else:
            y_true = epoch_y_true
        if epoch_y_au_score is None:
            y_au_score = self.y_au_score.detach().cpu().numpy()
        else:
            y_au_score = epoch_y_au_score

        metrics = {}

        if self.eval_type == "mae":
            # Compute Mean Absolute Error treating predictions as continuous.
            mae = mean_absolute_error(y_true, y_score)
            metrics["mae"] = mae

        else:
            # For classification tasks, first apply threshold to get binary predictions.
            y_pred_bin = self.apply_threshold(y_score)

            if self.eval_type == "binary":
                # When there is a single label per sample.
                acc = accuracy_score(y_true.ravel(), y_pred_bin.ravel())
                metrics["accuracy"] = acc
                metrics["hamming_loss"] = hamming_loss(y_true, y_pred_bin)
                metrics["precision"] = precision_score(y_true, y_pred_bin, average='binary', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred_bin, average='binary', zero_division=0)
                metrics["f1_score"] = f1_score(y_true, y_pred_bin, average='binary', zero_division=0)
                try:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_score)
                except ValueError:
                    metrics["auc_roc"] = None
                # Print confusion matrix
                cm = confusion_matrix(y_true, y_pred_bin)
                print("Confusion matrix:\n", cm)

            elif self.eval_type == "multilabel":
                # Compute metrics per label and then average.
                acc = 0
                for label in range(y_true.shape[1]):
                    label_acc = accuracy_score(y_true[:, label], y_pred_bin[:, label])
                    acc += label_acc
                metrics["accuracy"] = acc / y_true.shape[1]
                metrics["hamming_loss"] = hamming_loss(y_true, y_pred_bin)
                metrics["precision"] = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
                metrics["f1_score"] = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)

                auc = 0
                valid_labels = 0
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:
                        try:
                            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                            auc += label_auc
                            valid_labels += 1
                        except ValueError:
                            print(f"Label {i}: AUC computation failed.")
                    else:
                        print(f"Skipping label {i}: only one class present.")
                metrics["auc_roc"] = auc / valid_labels if valid_labels > 0 else 0.0

                # Print confusion matrices per label.
                for i in range(y_true.shape[1]):
                    y_true_label = y_true[:, i]
                    y_pred_label = y_pred_bin[:, i]
                    cm = confusion_matrix(y_true_label, y_pred_label)
                    print(f"Confusion matrix for label {i}:\n{cm}")

        # Add the mean aleatoric uncertainty to the metrics.
        metrics["aleatoric"] = float(np.mean(y_au_score))
        self.epoch_metrics = metrics
        return metrics

    
    def print_metrics(self):
        m = self.epoch_metrics
        if self.eval_type == "mae":
            print(f"MAE: {m['mae']:.4f}\tAleatoric Uncertainty: {m['aleatoric']:.5f}")
        else:
            print(f"ROC AUC: {m.get('auc_roc', 0):.4f}", end='\t')
            print(f"Accuracy: {m.get('accuracy', 0):.4f}", end='\t')
            print(f"F1 Score: {m.get('f1_score', 0):.4f}", end='\t')
            print(f"Recall: {m.get('recall', 0):.4f}", end='\t')
            print(f"Precision: {m.get('precision', 0):.4f}", end='\t')
            print(f"Hamming Loss: {m.get('hamming_loss', 0):.4f}", end='\t')
        print(f"Aleatoric Uncertainty: {m.get('aleatoric', 0):.5f}")

import torch
import torch.nn as nn
import torch.optim as optim
# from src.evaluations.monte_carlo import MultiLabelEvaluator as Evaluator
from src.utils.filename_manager import FilenameManager
from src.utils.results_writer import MetricsTracker
from src.utils.losses import BNN_BCEWithLogitsLoss, BNN_CrossEntropyLoss
from tqdm import tqdm

from src.utils.losses import heteroscedastic_loss

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

        self.evaluation_metrics = MultiLabelEvaluator(eval_type="mae")
        self.log_file_path = FilenameManager().get_filename('training_log')
        self.results_writer = MetricsTracker(self.log_file_path)

    def _initialize_optimizer(self):
        if self.optimizer_config == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config}")

    
    def _initialize_loss_function(self):
        return heteroscedastic_loss
        # if self.loss_function_config == "bnn_cross_entropy":
        #     return BNN_CrossEntropyLoss
        # elif self.loss_function_config == "bnn_binary_cross_entropy":
        #     print("BNN_CrossEntropyLoss")
        #     return BNN_CrossEntropyLoss
        # elif self.loss_function_config == "nn.CrossEntropyLoss":
        #     return nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(f"Unsupported loss function: {self.loss_function_config}")

    def train(self, val_loader):

        for epoch in range(1, self.num_epochs+1):
            print(f"Epoch [{epoch}/{self.num_epochs}]")

            train_loss, train_metrics = self.train_epoch(epoch=epoch)
            val_loss, val_metrics = self.validate_epoch(val_loader=val_loader)

            model_saved_path = FilenameManager().generate_model_filename(epoch=epoch, learning_rate=self.learning_rate, extension='pth')            
            self.model.save_model(model_saved_path)

            self.results_writer.update(epoch=epoch, batch=None, train_loss=train_loss, val_loss=val_loss, train_metrics=train_metrics, val_metrics=val_metrics)
            self.results_writer.save()
            print('Male test')
            male_monte_carlo = MonteCarloPredictionRegression(model=model, dataloader=male_test_loader, N=N_MonteCarloSimulation)
            male_monte_carlo.run_predictions()

            print('Female test')
            female_monte_carlo = MonteCarloPredictionRegression(model=model, dataloader=female_test_loader, N=N_MonteCarloSimulation)
            female_monte_carlo.run_predictions()


    def train_epoch(self, epoch):
        self.evaluation_metrics.reset_metrics()
        self.model.train()

        running_loss = 0.0

        for batch, (images, y)  in enumerate(tqdm(self.dataloader, desc="Processing Training Batches")):
            images, y = images.to(self.device), y.to(self.device)
            if epoch == 1 and batch == 0:
                print(f'{images.shape}, {y.shape}')

            self.model.train()
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred, var = self.model(images)
            loss = self.loss_function(pred, y, var)
            running_loss += loss.item()
            if epoch == 1 and batch == 0:
                print(f"model output dims: pred: {pred.shape} variance: {var.shape} true: {y.shape}")
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
        self.model.train()
        self.evaluation_metrics.reset_metrics()

        running_loss = 0.0
        with torch.no_grad():
            for batch, (images, y)  in enumerate(tqdm(val_loader, desc="Processing Validation Batches")):
                images, y = images.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {y.shape}')

                # Compute prediction and loss
                pred, var = self.model(images)
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


import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

class MonteCarloPredictionRegression:
    def __init__(self, model, dataloader, N=100):
        """
        Args:
            model: A heteroscedastic regression model that returns (predicted_mean, log_variance).
            dataloader: Dataloader for the test set.
            N (int): Number of Monte Carlo forward passes.
        """
        self.model = model
        self.dataloader = dataloader
        self.N = N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # For regression, we use a simple evaluation (e.g. MAE)

    def get_prediction(self, images):
        """
        Obtain a single prediction from the model.
        
        Returns:
            y_pred: Predicted mean as a NumPy array with shape (batch, 1, 1)
            y_var: Predicted variance (obtained from exp(log_var)) with shape (batch, 1, 1)
        """
        self.model.train()  # Activate dropout for MC sampling.
        with torch.no_grad():
            y_pred, y_log_var = self.model(images)

        # Detach and convert to NumPy, then add a new axis for MC sampling stacking.
        y_pred = y_pred.detach().cpu().numpy()  # shape: (batch, 1)
        y_log_var = y_log_var.detach().cpu().numpy()  # shape: (batch, 1)
        y_var = np.exp(y_log_var)
        return y_pred, y_var

    def run_predictions(self):
        """
        Runs MC sampling over the entire test set, computing:
            - Mean prediction (MC average)
            - Epistemic uncertainty (variance of predictions)
            - Aleatoric uncertainty (average predicted variance)
        Also computes the Mean Absolute Error (MAE).
        
        Returns:
            y_true_all, y_pred_all, aleatoric_all, epistemic_all
        """
        y_true_all = []
        y_pred_all = []
        epistemic_all = []
        aleatoric_all = []

        # Activate dropout for MC sampling.
        for batch_data in tqdm(self.dataloader):
            images, y = batch_data
            images = images.to(self.device)
            batch_size = images.shape[0]
            # Containers to store MC samples (shape: [batch, N, 1])
            y_pred_samples = np.empty((batch_size, self.N, 1))
            y_var_samples = np.empty((batch_size, self.N, 1))

            for i in range(self.N):
                y_pred, y_var = self.get_prediction(images)
                # y_pred and y_var are of shape (batch, 1); squeeze the extra dimension.
                y_pred_samples[:, i, :] = y_pred
                y_var_samples[:, i, :] = y_var

            # Compute the Monte Carlo mean prediction.
            y_pred_mean = np.mean(y_pred_samples, axis=1)  # (batch, 1)
            # Epistemic uncertainty is the variance across the N predictions.
            epi_unc = np.var(y_pred_samples, axis=1)  # (batch, 1)
            # Aleatoric uncertainty is the average predicted variance.
            alea_unc = np.mean(y_var_samples, axis=1)  # (batch, 1)

            y_true_all.extend(y.detach().cpu().numpy())
            y_pred_all.extend(y_pred_mean)
            epistemic_all.extend(epi_unc)
            aleatoric_all.extend(alea_unc)

        # Concatenate results from all batches.
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        epistemic_all = np.concatenate(epistemic_all, axis=0)
        aleatoric_all = np.concatenate(aleatoric_all, axis=0)
        print(f"Dimension of the testing set {y_true_all.shape}{y_pred_all.shape}{epistemic_all.shape}{aleatoric_all.shape}")
        # Compute evaluation metric: MAE.
        mae = mean_absolute_error(y_true_all, y_pred_all)
        print(f"MAE: {mae:.4f}")

        # Compute average uncertainties.
        avg_epi = np.mean(epistemic_all)
        avg_alea = np.mean(aleatoric_all)
        print(f"Average Epistemic Uncertainty: {avg_epi}")
        print(f"Average Aleatoric Uncertainty: {avg_alea}")

        return mae, avg_alea, avg_epi

        # return y_true_all, y_pred_all, aleatoric_all, epistemic_all

# Example usage:
# Assuming `model_reg` is your trained heteroscedastic regression model,
# and `test_loader` is your test DataLoader.
# predictor = MonteCarloPredictionRegression(model_reg, test_loader, N=100)
# y_true, y_pred, alea_unc, epi_unc = predictor.run_predictions()

from src.utils.config_reader import ConfigReader
from src.dataloader.dataloader_factory import dataloader_factory
from src.models.model_factory import model_factory
from src.evaluations.monte_carlo import MonteCarloPrediction

import argparse
if __name__ == "__main__":
    import os
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a specified dataset.")
    parser.add_argument('--model', default='UTKFaceAgeModel',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--dataset', default='UTKFace', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')
    # parser.add_argument('--dataset', default='CheXpert', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')
    parser.add_argument('--task', default='train_bnn',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    # parser.add_argument('--task', default='check_medvit_test',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    # parser.add_argument('--task', default='fine_tune_biggan',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--task_config', default='medvit_test',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')

    args = parser.parse_args()
    print(args)


    dataset_name = args.dataset
    model_name = args.model
    task_name = args.task
    task_config_name = args.task_config

    print(args)
    
    file_manger = FilenameManager(model_name=model_name, dataset_name=dataset_name, task_name=task_name)

    # Load configuration files
    configs = ConfigReader().load_all_configs()
    datasets_config = configs['datasets']
    models_config = configs['models']
    config = configs['project']

    # Device configuration (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Initializing DataLoader for {dataset_name}...")
    # Initialize DataLoader for the specified dataset
    dataset_info = datasets_config['datasets'].get(dataset_name)
    print(dataset_info)
    print(dataset_info['train']['img_dim'])
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
    
    # Create the DataLoaders for train, validation, and test splits
    train_loader = dataloader_factory(dataset_name, 'train', dataset_info)
    val_loader = dataloader_factory(dataset_name, 'val', dataset_info)

    male_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group=0)
    female_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group=1)
    print("Model Config", models_config)
    model = UTKFaceAgeModel(task='regression', drop_rate=0.5, hidden_layer=128)
    model_saved_location = "outputs/train_bnn_UTKFaceAgeModel_UTKFace_20250224_121334/models/model_weights_epoch_50_lr_0.005_20250224_121334.pth"
    # task_config['bnn_model_location']
    model.load_model(model_saved_location) 
    # model = model_factory(model_name=model_name, models_config=models_config)
    print(model)

    N_MonteCarloSimulation = config['N_MonteCarloSimulation']
    
    if task_name == 'train_bnn':

        # Initialize Trainer
        print(f"Training {model_name} on {dataset_name}...")
        trainer = Trainer(
            model=model,
            dataloader=train_loader,
            config=config
        )

        # Train the model
        trainer.train(val_loader)

    elif task_name == 'test_bnn':
        # task_config = config[task_name][task_config_name]
        # print(task_config)

        model_saved_location = "outputs/train_bnn_UTKFaceAgeModel_UTKFace_20250224_121334/models/model_weights_epoch_50_lr_0.005_20250224_121334.pth"
        # task_config['bnn_model_location']
        model.load_model(model_saved_location)        
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        print('Male test')
        male_monte_carlo = MonteCarloPredictionRegression(model=model, dataloader=male_test_loader, N=N_MonteCarloSimulation)
        male_monte_carlo.run_predictions()

        print('Female test')
        female_monte_carlo = MonteCarloPredictionRegression(model=model, dataloader=female_test_loader, N=N_MonteCarloSimulation)
        female_monte_carlo.run_predictions()

