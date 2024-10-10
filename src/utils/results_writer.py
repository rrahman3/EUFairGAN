import pandas as pd

class MetricsTracker:
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        self.metrics = {
            "epoch": [],
            "batch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }

    def update(self, epoch, batch, train_loss, val_loss, train_accuracy, val_accuracy):
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["val_accuracy"].append(val_accuracy)

    def save(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_file, index=False)
        print(f"Metrics saved to {self.metrics_file}")
