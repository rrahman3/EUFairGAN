# evaluations/evaluator.py

import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from math import log10
import torch.nn.functional as F
import torchvision.transforms as transforms

class Evaluator:
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

    def update_metrics(self, y_true, y_pred, y_variance): # y_ture is the binary form of the y_true (None, 1, num_classes) # [[0, 0, 1, 0]] (1, 1, 4)

        # Get predictions
        pred_softmax = F.softmax(y_pred, dim=1)

        _, y_pred_argmax = torch.max(pred_softmax, 1)
        _, y_true_argmax = torch.max(y_true, 1)

        self.all_labels_direct.extend(y_true_argmax.cpu().numpy())
        self.all_predictions_driect.extend(y_pred_argmax.cpu().numpy())

        self.all_labels_binary.extend(y_true.detach().cpu().numpy())
        self.all_predictions_probabilities.extend(pred_softmax.detach().cpu().numpy())
        
        self.all_varience.extend(y_variance.detach().cpu().numpy())

        # epoch_variance += torch.mean(y_variance) * images.shape[0]

        # epoch_test_loss = epoch_test_loss / self.len
        # epoch_variance = epoch_variance / self.len
        # y_pred_argmax = y_pred_argmax.detach().cpu().numpy()
        # y_true_argmax = y_true_argmax.detach().cpu().numpy()

        # y_pred = y_pred.detach().cpu().numpy()
        # y_true = y_true.detach().cpu().numpy()



    def compute_epoch_metrics(self):
        """Combine batch metrics to get epoch metrics."""

        self.auc_scores = roc_auc_score(self.all_labels_binary, self.all_predictions_probabilities, multi_class='ovo')
        
        self.f1_scores = f1_score(self.all_labels_direct, self.all_predictions_driect, average='macro')
        
        self.accuracies = accuracy_score(self.all_labels_direct, self.all_predictions_driect)
        
        self.conf_matrices = confusion_matrix(self.all_labels_direct, self.all_predictions_driect)

        self.epoch_metrics = {
            'variance': np.mean(self.all_varience),
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
        print(f"Confusion Metrics\n{self.epoch_metrics['confusion_metrix']}", end="\n\n")


epsilon = 0.000001
class MonteCarloPrediction:
    def __init__(self, model, dataloader, T=100):
        self.model = model
        self.dataloader = dataloader
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def asfsdgd(self):
        with torch.no_grad():
            for batch, (images, genders, y)  in enumerate(self.dataloader):
                images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {genders.shape}, {y.shape}')

                # Compute prediction and loss
                for i in range(1, self.T+1):
                    pred, var = self.model(images, genders)
                    loss = self.loss_function(y, pred, var)
                    running_loss += loss.item()

                    self.evaluation_metrics.update_metrics(y_true=y, y_pred=pred, y_variance=var)

            
        epoch_loss = running_loss / len(self.val_loader)

    def montecarlo_prediction(self):
        y_pred_T = np.array([model.predict(test.sequences, verbose=0) for _ in range(T)]) #(T, batch_size, (num_items+softplus))
        # print(predictions.shape)
        y_pred = np.mean(y_pred_T, axis=0) #(T, batch_size, (num_items+softplus))
        # print(prediction_probabilities.shape)
        y_pred = y_pred[:, :-1] #(batch_size, num_items)
        # print(predictions.shape)
        softmax_y_pred = tf.nn.softmax(y_pred)
        epistemic_uncertainty = np.apply_along_axis(predictive_entropy, axis=1, arr=softmax_y_pred) #(batch_size, 1), varience of each user
        # print(prediction_variances.shape)
        m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)
        m.update_state(test.targets, softmax_y_pred)

        #return HR@20, Average epistemic Uncertainty, array of epistemic uncertainty for each user
        #Similat to evaluate_model() method
        return m.result().numpy(), np.mean(epistemic_uncertainty), epistemic_uncertainty

    def predictive_entropy(self, prob):
        return -1 * np.sum(np.log(prob+epsilon) * (prob))