import unittest
from src.utils.config_reader import ConfigReader

class TestConfigReader(unittest.TestCase):
    def setUp(self):
        """Set up the ConfigReader instance."""
        self.config_reader = ConfigReader(base_path='configs')

    def test_load_config(self):
        """Test loading a single configuration file."""
        config = self.config_reader.load_config('config.yaml')
        self.assertIn('training', config)
        self.assertIn('num_epochs', config['training'])
        self.assertIn('learning_rate', config['training'])

        self.assertEqual(config['training']['num_epochs'], 100)
        self.assertEqual(config['training']['learning_rate'], 0.0001)

    def test_load_all_configs(self):
        """Test loading all configuration files."""
        configs = self.config_reader.load_all_configs()
        self.assertIn('project', configs)
        self.assertIn('datasets', configs)
        self.assertIn('models', configs)

    def test_invalid_config(self):
        """Test loading a non-existent configuration file."""
        with self.assertRaises(FileNotFoundError):
            self.config_reader.load_config('invalid_config.yaml')


from src.dataloader.medical_dataset import NIHChestXrayDataset
class TestDataset(unittest.TestCase):
    def test_nihcc_chest_xray():
        dataset = NIHChestXrayDataset(
            image_dir="data/nihcc_chest_xray/images/",
            metadata_file="data/nihcc_chest_xray/miccai2023_nih-cxr-lt_labels_train.csv",
            image_dim=(224, 224),
        )
        print(len(dataset))

if __name__ == '__main__':
    unittest.main()
