# utils/filename_manager.py

import datetime
import os

class FilenameManager:
    _instance = None

    def __new__(cls, model_name=None, dataset_name=None, task_name=None):
        if cls._instance is None:
            cls._instance = super(FilenameManager, cls).__new__(cls)
            cls._instance.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._instance.base_folder = f"{task_name}_{model_name}_{dataset_name}_{cls._instance.timestamp}"
            # Initialize with default filenames and directories
            cls._instance.filenames = {
                "training_log": f"outputs/{cls._instance.base_folder}/training_log.csv",
                "validation_log": f"outputs/{cls._instance.base_folder}/validation_log.csv",
                "evaluation_log": f"outputs/{cls._instance.base_folder}/evaluations.csv",
                "images": f"outputs/{cls._instance.base_folder}/images/",
                "models": f"outputs/{cls._instance.base_folder}/models/"
            }
            os.makedirs('outputs', exist_ok=True)
            os.makedirs(f"outputs/{cls._instance.base_folder}", exist_ok=True)
            os.makedirs(f"outputs/{cls._instance.base_folder}/images", exist_ok=True)
            os.makedirs(f"outputs/{cls._instance.base_folder}/models", exist_ok=True)
        return cls._instance

    def get_filename(self, key: str) -> str:
        """Get the filename associated with the given key."""
        return self.filenames.get(key, None)

    def set_filename(self, key: str, filename: str) -> None:
        """Set a new filename for the given key."""
        self.filenames[key] = filename


    def generate_model_filename(self, epoch=None, batch_size=None, learning_rate=None, extension='pth'):
        filename = self.get_filename('models') + '/model_weights_'

        if epoch is not None:
            filename += f"_epoch{epoch}"
        if batch_size is not None:
            filename += f"_batch{batch_size}"
        if learning_rate is not None:
            filename += f"_lr{learning_rate}"

        filename += f"_{self.timestamp}.{extension}"
        return filename

    def generate_filename(self, filename, epoch=None, batch_size=None, learning_rate=None, extension='csv'):

        if epoch is not None:
            filename += f"_epoch{epoch}"
        if batch_size is not None:
            filename += f"_batch{batch_size}"
        if learning_rate is not None:
            filename += f"_lr{learning_rate}"

        filename += f"_{self.timestamp}.{extension}"
        return filename
