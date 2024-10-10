import os
import yaml

class ConfigReader:
    def __init__(self, base_path='configs'):
        self.base_path = base_path

    def load_config(self, config_filename):
        config_file_path = os.path.join(self.base_path, config_filename)
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_all_configs(self):
        # Load all necessary configuration files
        config = {}
        config['project'] = self.load_config('config.yaml')
        config['datasets'] = self.load_config('datasets.yaml')
        config['models'] = self.load_config('models.yaml')
        return config
