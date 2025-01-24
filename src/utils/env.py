# env.py


import importlib.util
import os
import sys
from dotenv import load_dotenv, dotenv_values


class EnvLoader:
    def __init__(self):
        """
        Initialize the EnvLoader class and prepare attributes for storing the .env path and loaded values.
        """
        # Resolve the path to the .env file relative to this file's location
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.env_path = os.path.join(self.base_dir, "config.env")
        self.env_values = {}
        
        # Automatically load the .env file
        self.load()
        
    def load_config_module(self, module_path):
        """
        Dynamically import a module from the given file path.

        Args:
            module_path (str): The path to the Python file containing the config.

        Returns:
            dict: The `config` object from the imported module.
        """
        if not os.path.isfile(module_path):
            raise FileNotFoundError(f"Config file not found: {module_path}")
        
        # Create a module spec
        spec = importlib.util.spec_from_file_location("config_module", module_path)
        config_module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(config_module)
        
        # Return the 'config' dictionary from the module
        if not hasattr(config_module, "config"):
            raise AttributeError(f"The module at {module_path} does not contain a 'config' attribute.")
        
        return config_module.config

    def load(self):
        """
        Load the .env file, store its contents in the env_values attribute, and add PYTHONPATH to sys.path.

        Returns:
            dict: A dictionary of the loaded .env file's contents.
        """
        # Check if the .env file exists
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path)  # Load into os.environ
            self.env_values = dotenv_values(self.env_path)  # Load into a dictionary for reference

            # Add PYTHONPATH to sys.path if defined
            python_path = self.env_values.get("PYTHONPATH")
            if python_path and python_path not in sys.path:
                sys.path.append(python_path)
            return self.env_values
        else:
            raise FileNotFoundError(f".env file not found at: {self.env_path}")
    
    def get(self, key):
        """
        Load the env data from the provided key.
        
        Args:
            key (str): The key to look up in the environment variables.
        
        Returns:
            value: The provided key's value.
            
        Standard keys:
            DATABASE
        """        
        if key in self.env_values:
            value = self.env_values.get(key)
            return value
        else:
            raise KeyError(f"Key '{key}' not found in the environment variables")
            