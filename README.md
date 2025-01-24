"model_development" is a framework for training timeseries neural networks. Development is accompanied 
by a dummy project which helps with testing functionality and outlining requirments for practitioners.

Quick start:
Pull the repository and configure your model training workflow by filling in the config, .env, and 
project-specific functions in the /test folder files. 
   - .env file contains database paths
   - config.py files specify function attributes (module paths, model architecture, process arguements, etc.)
   - custom functions and their required format are in the test folder .py files for each high-level function
Create a project directory with a 'data' folder for your database(s). Organization should be based on the base
feature creation data. If you think the features are appliable to many iterations then creating a "models"
folder in the project directory will help with organization. The workflow requires saving model configuration
data for loading and making inferences after training. Ensure your project file is segmented so files are
properly associated with one another. Many workflows will benefit from storing the iterables in a containerized
way. 

Example project directory:
/Project1
  /data
    /.db
    /features.py
    /targets.py
  /models
    /model1
      /weights
      /modules.py (however many needed)

Requirements:
1. Create and populate your feature/target database.
2. Create custom data preprocessing and model training modules
3. Setup .env file and the function's config.py (found in each /src folder)
