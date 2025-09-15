import json

class ConfigLoader:
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)


