from src.evaluations.evaluator import Evaluator
import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm

epsilon = 0.000001
class MonteCarloPrediction:
    def __init__(self, model, dataloader, N=100):
        self.model = model
        self.dataloader = dataloader
        self.N = N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_metrics = MultiLabelEvaluator()
        self.task = 'multi=label'
        self.data_len = next(iter(self.dataloader))

    def get_prediction(self, images, genders):
        if genders==None:
            y_pred, y_var = self.model(images)
        else:
            y_pred, y_var = self.model(images, genders)

        # y_pred = y_pred.detach().cpu().numpy()
        # y_pred = y_pred[:, np.newaxis, :]

        # y_var = y_var.detach().cpu().numpy()
        # y_var = y_var[:, np.newaxis, :]

        return y_pred, y_var


    def asfsdgd(self):
        y_true = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)
        y_au_score = torch.tensor([]).to(self.device)

        with torch.no_grad():
            num_batch = 0
            for batch_data  in tqdm(self.dataloader):

                if len(batch_data) == 2:
                    images, y = batch_data
                    images, y = images.to(self.device), y.to(self.device)
                    y = y.float()
                    genders = None
                    if num_batch == 0:
                        print(f'{images.shape}, {y.shape}')
                elif len(batch_data) == 3:
                    images, genders, y = batch_data
                    images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                    if num_batch == 0:
                        print(f'{images.shape}, {y.shape}')
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch_data)}")
            
                self.model.train()

                # Compute prediction and loss
                y_pred_N = np.empty((images.shape[0], self.N, y.shape[-1])) #(batch_size, N, num_classes)
                y_var_N = np.empty((images.shape[0], self.N, 1))

                for i in range(1, self.N+1):
                    y_pred, y_var = self.get_prediction(images=images, genders=genders) #(batch_size, 1, num_classes)

                    # Store the predictions and variances at the current iteration index (i-1)
                    y_pred_N[:, i - 1, :] = y_pred[:, 0, :]  # Using [0] to get rid of the added dimension
                    y_var_N[:, i - 1, :] = y_var[:, 0, :]



                y_pred_mean = np.mean(np.array(y_pred_N), axis=1) #(batch, num_classes)
                if self.task == 'multi-label':
                    y_pred_softmax = torch.sigmoid(y_pred_mean, axis=1)
                else:
                    y_pred_softmax = softmax(y_pred_mean, axis=1)

                y_true = torch.cat((y_true, y_pred_mean), 0)
                y_score = torch.cat((y_score, y_pred_softmax), 0)

                y_var_mean = np.mean(np.array(y_var_N), axis=1) #(batch, 1)
                y_au_score = torch.cat((y_au_score, y_var_mean), 0)


                # # update evaluation metrics
                # self.evaluation_metrics.update_metrics(y_true=y, y_pred=y_pred_softmax)
                # self.evaluation_metrics.update_variance(variance=aleatoric_uncertainty)
                # self.evaluation_metrics.update_entropy(entropy=epistemic_uncertainty)

            # epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
            # self.evaluation_metrics.print_metrics()
            epistemic_uncertainty = np.apply_along_axis(self.predictive_entropy, axis=1, arr=y_score) #(batch_size, 1), epistemic uncertainty of each user
            aleatoric_uncertainty = np.mean(y_au_score) #(batch, 1)
            print(f'"Aleatoric Uncertainty":{aleatoric_uncertainty}\n"Epistemic Uncertainty":{epistemic_uncertainty}')
            result = self.evaluation_metrics.evaluate(y_true=y_true, y_score=y_score)
            print(result)

    def predictive_entropy(self, prob):
        return -1 * np.sum(np.log(prob+epsilon) * (prob)) # entropy of class distribution
    

# evaluations/evaluator.py

import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, hamming_loss
import numpy as np
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class MultiLabelEvaluator:
    def __init__(self, n_labels=14, threshold=0.5):
        self.n_labels = n_labels  # Number of labels per sample
        self.threshold = threshold  # Threshold for binary classification

    def apply_threshold(self, y_pred):
        """
        Apply threshold to the predicted probabilities to convert to binary labels.
        """
        return (y_pred >= self.threshold).astype(int)

    def evaluate(self, y_pred, y_true):
        """
        Evaluate model performance using various multi-label classification metrics.
        """
        # Apply threshold to convert probabilities to binary labels
        y_pred_bin = self.apply_threshold(y_pred)
        
        # Accuracy (Subset accuracy): exact match of all labels
        accuracy = accuracy_score(y_true, y_pred_bin)

        # y_true = y_true.squeeze()
        # y_pred = y_pred.squeeze()

        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pred_bin[:, label])
            acc += label_acc
        accuracy = acc / y_true.shape[1]
            
        
        # Hamming loss: fraction of labels that are incorrectly predicted
        h_loss = hamming_loss(y_true, y_pred_bin)
        
        # Precision, Recall, F1 Score (Macro average across labels)
        precision = precision_score(y_true, y_pred_bin, average='macro')
        recall = recall_score(y_true, y_pred_bin, average='macro')
        f1 = f1_score(y_true, y_pred_bin, average='macro')
        
        # AUC-ROC (per label, then averaged)
        auc = 0
        for i in range(y_true.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc += label_auc
        auc_roc = auc / y_true.shape[1]
        # auc_roc = roc_auc_score(y_true, y_pred, average='macro')
        
        # Return all metrics as a dictionary
        return {
            "accuracy": accuracy,
            "hamming_loss": h_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc
        }
    
class MonteCarloEvaluator:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics for a new epoch"""
        self.f1_scores = []
        self.auc_scores = []
        self.accuracies = []
        self.conf_matrices = []
        
        self.all_labels_direct = [] #contains the actual class labels , 1, 3, 4
        self.all_labels_binary = [] # contains in binary form [0, 0, 1, 0] for true class 2 #shpe=(None, 1, num_classes)
        self.all_predictions_probabilities = [] #contains all the classes probability
        self.all_predictions_driect = [] #contains the maxminum probaility class
        self.all_varience = []
        self.all_entropy = []

    def update_metrics(self, y_true, y_pred): # y_ture is the binary form of the y_true (None, 1, num_classes) # [[0, 0, 1, 0]] (1, 1, 4)
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        y_true_argmax = np.argmax(y_true, axis=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        
        self.all_labels_direct.extend(y_true_argmax)
        self.all_predictions_driect.extend(y_pred_argmax)

        self.all_labels_binary.extend(y_true)
        self.all_predictions_probabilities.extend(y_pred)
    
    def update_variance(self, variance):
        self.all_varience.extend(variance)

    def update_entropy(self, entropy):
        self.all_entropy.extend(entropy)

    def compute_epoch_metrics(self):
        """Combine batch metrics to get epoch metrics."""

        self.auc_scores = roc_auc_score(self.all_labels_binary, self.all_predictions_probabilities, multi_class='ovo')
        
        self.f1_scores = f1_score(self.all_labels_direct, self.all_predictions_driect, average='macro')
        
        self.accuracies = accuracy_score(self.all_labels_direct, self.all_predictions_driect)
        
        self.conf_matrices = confusion_matrix(self.all_labels_direct, self.all_predictions_driect)

        self.epoch_metrics = {
            'variance': np.mean(self.all_varience),
            'entropy': np.mean(self.all_entropy),
            'accuracy': self.accuracies,
            'f1_score': self.f1_scores,
            'auc': self.auc_scores,
            'confusion_metrix': self.conf_matrices
        }
        return self.epoch_metrics

    def print_metrics(self):
        print(f"F1 Score: {self.epoch_metrics['f1_score']:.4f}", end='\t')
        print(f"AUC: {self.epoch_metrics['auc']:.4f}", end='\t')
        print(f"Accuracy: {self.epoch_metrics['accuracy']:.4f}", end='\t')
        print(f"Aleatoric Uncertainty: {self.epoch_metrics['variance']:.5f}", end='\n')
        print(f"Epistemic Uncertainty: {self.epoch_metrics['entropy']:.5f}", end='\n')
        print(f"Confusion Metrics\n{self.epoch_metrics['confusion_metrix']}", end="\n\n")


