import pandas as pd
import os

class MetricsTracker:
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        self.metrics = {
            "epoch": [],
            "batch": [],
            "train_loss": [],
            "val_loss": []
        }

    def update(self, epoch, batch, train_loss, val_loss, train_metrics, val_metrics):
        self.metrics["epoch"].append(epoch)
        self.metrics["batch"].append(batch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)

        # Dynamically append train_metrics and val_metrics
        for key, value in train_metrics.items():
            metric_key = f"train_{key}"
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []  # Create a new list for this metric
            self.metrics[metric_key].append(value)

        for key, value in val_metrics.items():
            metric_key = f"val_{key}"
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []  # Create a new list for this metric
            self.metrics[metric_key].append(value)


    def save(self):
        new_data = pd.DataFrame(self.metrics)

        # Check if the file exists
        if os.path.exists(self.metrics_file):
            # If the file exists, read it into a DataFrame
            existing_data = pd.read_csv(self.metrics_file)

            # Append the new data to the existing data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            # If the file does not exist, use the new data as is
            combined_data = new_data

        # Save the combined data back to the CSV file
        combined_data.to_csv(self.metrics_file, index=False)
        print(f"Metrics saved to {self.metrics_file}")
