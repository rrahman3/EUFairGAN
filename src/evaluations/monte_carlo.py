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

        

        y_pred = y_pred.detach().cpu().numpy()
        y_pred = y_pred[:, np.newaxis, :]

        y_var = y_var.detach().cpu().numpy()
        y_var = y_var[:, np.newaxis, :]

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
                
                num_batch += 1
            
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

                y_true = torch.cat((y_true, torch.tensor(y, device=y_true.device)), 0)
                y_score = torch.cat((y_score, torch.tensor(y_pred_softmax, device=y_score.device)), 0)

                y_var_mean = np.mean(np.array(y_var_N), axis=1) #(batch, 1)
                y_au_score = torch.cat((y_au_score, torch.tensor(y_var_mean, device=y_au_score.device)), 0)


                # # update evaluation metrics
                # self.evaluation_metrics.update_metrics(y_true=y, y_pred=y_pred_softmax)
                # self.evaluation_metrics.update_variance(variance=aleatoric_uncertainty)
                # self.evaluation_metrics.update_entropy(entropy=epistemic_uncertainty)

            # epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
            # self.evaluation_metrics.print_metrics()
            y_score = y_score.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            y_au_score = y_au_score.detach().cpu().numpy()

            epistemic_uncertainty = np.apply_along_axis(self.predictive_entropy, axis=1, arr=y_score) #(batch_size, 1), epistemic uncertainty of each user
            aleatoric_uncertainty = np.mean(y_au_score) #(batch, 1)
            epistemic_uncertainty = np.mean(epistemic_uncertainty)
            print(f'"Aleatoric Uncertainty":{aleatoric_uncertainty}\n"Epistemic Uncertainty":{epistemic_uncertainty}')
            result = self.evaluation_metrics.compute_epoch_metrics(y_pred=y_score, y_true=y_true)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_labels = n_labels  # Number of labels per sample
        self.threshold = threshold  # Threshold for binary classification
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
        Evaluate model performance using various multi-label classification metrics.
        """
        if epoch_y_pred is None:
            self.y_score = self.y_score.detach().cpu().numpy()
        if epoch_y_true is None:
            self.y_true = self.y_true.detach().cpu().numpy()
        if epoch_y_au_score is None:
            self.y_au_score = self.y_au_score.detach().cpu().numpy()

        # Apply threshold to convert probabilities to binary labels
        y_pred_bin = self.apply_threshold(self.y_score)
        
        # # Accuracy (Subset accuracy): exact match of all labels
        # accuracy = accuracy_score(y_true, y_pred_bin)

        # y_true = y_true.squeeze()
        # y_pred = y_pred.squeeze()

        acc = 0
        for label in range(self.y_true.shape[1]):
            label_acc = accuracy_score(self.y_true[:, label], y_pred_bin[:, label])
            acc += label_acc
        self.accuracy = acc / self.y_true.shape[1]
            
        
        # Hamming loss: fraction of labels that are incorrectly predicted
        self.h_loss = hamming_loss(self.y_true, y_pred_bin)
        
        # Precision, Recall, F1 Score (Macro average across labels)
        self.precision = precision_score(self.y_true, y_pred_bin, average='macro')
        self.recall = recall_score(self.y_true, y_pred_bin, average='macro')
        self.f1 = f1_score(self.y_true, y_pred_bin, average='macro')
        
        # AUC-ROC (per label, then averaged)
        auc = 0
        for i in range(self.y_true.shape[1]):
            label_auc = roc_auc_score(self.y_true[:, i], self.y_score[:, i])
            auc += label_auc
        self.auc_roc = auc / self.y_true.shape[1]
        # auc_roc = roc_auc_score(y_true, y_pred, average='macro')

        print(f"--------------------------Confusion matrix------------------------------:")
        for i in range(self.y_true.shape[1]):
            y_true_label = self.y_true[:, i]
            y_pred_label = self.y_score[:, i]
            y_pred_label = (y_pred_label >= 0.5).astype('float')
            
            cm = confusion_matrix(y_true_label, y_pred_label)
            print(f"Confusion matrix for label {i}: \n{cm}")
        
        # Return all metrics as a dictionary
        self.epoch_metrics = {
            "accuracy": self.accuracy,
            "hamming_loss": self.h_loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1,
            "auc_roc": self.auc_roc,
            "aleatoric": np.mean(self.y_au_score) #(batch, 1)
        }
        return self.epoch_metrics
    
    def print_metrics(self):
        print(f"ROC AUC: {self.epoch_metrics['auc_roc']:.4f}", end='\t')
        print(f"Accuracy: {self.epoch_metrics['accuracy']:.4f}", end='\t')
        print(f"Aleatoric Uncertainty: {self.epoch_metrics['aleatoric']:.5f}", end='\n')


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


