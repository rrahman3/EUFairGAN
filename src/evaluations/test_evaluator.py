import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize   
from collections import namedtuple

Metrics = namedtuple('Metrics', [ 'loss','accuracy', 'f1_score', 'variance'])
TestMetrics = namedtuple('TestMetrics', [ 'loss','accuracy', 'f1_score', 'variance', 'auc'])


class Evaluator:
    def __init__(self, model, test_loader, loss_fn=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)

        self.test_loader = test_loader
        
        self.loss_fn = loss_fn       

        self.len = len(self.test_loader.dataset)


    def evaluate(self):
        self.model.train()
        with torch.no_grad():
            epoch_test_loss = 0.0
            epoch_variance = 0.0
            epoch_f1_scores = []
            all_labels_direct = [] #contains the actual class labels , 1, 3, 4
            all_labels_binary = [] # contains in binary form [0, 0, 1, 0] for true class 2
            all_predictions_probabilities = [] #contains all the classes probability
            all_predictions_driect = [] #contains the maxminum probaility class

            for step, (images, genders, y)  in enumerate(self.test_loader):  # val_loader should be defined
                images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                # print(f'image.shape {images.shape}, genders shape {genders.shape}, age shape {y.shape}')

                pred, var = self.model(images, genders)
                if self.loss_fn is not None:
                    test_loss = self.loss_fn(y, pred, var) #returns the mean loss
                    epoch_test_loss += test_loss.item() * images.shape[0]

                # Get predictions
                pred_softmax = F.softmax(pred, dim=1)

                _, y_pred = torch.max(pred_softmax, 1)
                _, y_true = torch.max(y, 1)

                all_labels_direct.extend(y_true.cpu().numpy())
                all_predictions_driect.extend(y_pred.cpu().numpy())

                all_labels_binary.extend(y.detach().cpu().numpy())
                all_predictions_probabilities.extend(pred_softmax.cpu().numpy())
                

                epoch_variance += torch.mean(var) * images.shape[0]

            epoch_test_loss = epoch_test_loss / self.len
            epoch_variance = epoch_variance / self.len

            epoch_auc = roc_auc_score(all_labels_binary, all_predictions_probabilities, multi_class='ovo')
            epoch_f1_score = f1_score(all_labels_direct, all_predictions_driect, average='macro')
            epoch_accuracy = accuracy_score(all_labels_direct, all_predictions_driect)
            epoch_conf_matrix = confusion_matrix(all_labels_direct, all_predictions_driect)

            test_metrics = TestMetrics(epoch_test_loss, epoch_accuracy, epoch_f1_score, epoch_variance.item(), epoch_auc)

            print(f"Loss: {test_metrics.loss:.5f}", end='\t')
            print(f"F1 Score: {test_metrics.f1_score:.4f}", end='\t')
            print(f"AUC: {test_metrics.auc:.4f}", end='\t')
            print(f"Accuracy: {test_metrics.accuracy:.4f}", end='\t')
            print(f"Aleatoric Uncertainty: {test_metrics.variance:.5f}", end='\n')
            print(f"Confusion Metrics\n{epoch_conf_matrix}", end="\n\n")

        return test_metrics

