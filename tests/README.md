# Test Framework for Model Training
## Overview
This repository contains a dummy project which uses a small dataset to replicate the original, real-world use case. Test each '/src' module by walking through the pipeline, from data preparation to training and evaluation. Copying the folders into the '/projects' directory will also help with quick starting a project.

## Future Development
Creating a separate '/templates' folder and modifying this directory to include integration and unit testing is more conventional and practical. 

### Suggested Structure:
Enabling users to train locally with an arbitrarily large dataset while enforcing documentation and best practices means being adaptable to different use cases and datasets. When modifying '/src' modules to accommodate new workflows, new 'dummy_tests' should be added to mimic its use. Contributors should expand the '/unit_tests' and '/integration_tests' as well.

### Modify the directory:
```sh
/tests
├── /dummy_tests                   # Stores feature-specific test projects
│   ├── /baseline_project           # Original test case (golden standard)
│   │   ├── /data_preparation.py
│   │   ├── /training.py
│   │   ├── /evaluation.py
│   │   ├── /results
│   │   └── README.md
│   │
│   ├── /feature_xyz_project        # A dummy test for a new feature
│   ├── /feature_abc_project        # Another dummy test for a different feature
│
├── /unit_tests                     # Unit tests for individual functions
├── /integration_tests               # Full pipeline tests
└── /fixtures                        # Reusable test data & configs
```

### Automate Running All Dummy Test Projects: 
Instead of requiring manual testing across all dummy projects to ensure full compatibility, create a script to automatically execute them.
Example 'run_dummy_tests.py':
```sh
import pytest
import subprocess
import os

DUMMY_TESTS_DIR = "tests/dummy_tests"

@pytest.mark.parametrize("project", [d for d in os.listdir(DUMMY_TESTS_DIR) if os.path.isdir(os.path.join(DUMMY_TESTS_DIR, d))])
def test_dummy_project(project):
    project_path = os.path.join(DUMMY_TESTS_DIR, project)
    
    for script in ["data_preparation.py", "training.py", "evaluation.py"]:
        script_path = os.path.join(project_path, script)
        if os.path.exists(script_path):
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            assert result.returncode == 0, f"{script} failed in {project}"

```
```sh
python tests/run_dummy_tests.py
```

## Current Setup:
*The original dummy project*

### Folder Structure
```sh
/test 
│── /data_preparation  
│   ├── config.py
│   ├── test_create_dummy_data_db.py  
│   ├── test_feature_engineering.py  
│   ├── test_target_engineering.py
│   └── test_data_preparation.py
│  
│── /training  
│   ├── /bad_keys
│   ├── /iterations
│   ├── /prepped_data
│   ├── /scalers
│   ├── /weights_dict
│   ├── config.py  
│   ├── test_process_raw_data.py  
│   ├── test_custom_train_funcs.py  
│   └── test_training.py
│  
│── /evaluation  
│   ├── /predictions
│   ├── config.py  
│   ├── test_custom_eval_funcs.py
│   └── test_evaluation.py
│  
│── /data
│── config.env
└── README.md  
```

### Detailed Module Breakdown
Data Preparation (/data_preparation)
Purpose: Stores raw data and applies feature engineering and target generation to prepare inputs for training.
Testing Focus: Ensures that features and targets are generated correctly and are aligned for model training.

### Training (/training)
Purpose: Handles data cleaning, model training, and hyperparameter tuning.
Testing Focus: Ensures that the model is trained using clean, correctly formatted data as expected.

### Evaluation (/evaluation)
Purpose: Loads trained models, generates predictions, and evaluates performance.
Testing Focus: Ensures that predictions are generated correctly, model explanations are valid, metrics are computed reliably, and model calibration is appropriate.

### How This Framework Supports Testing
- Step-by-Step Data Availability: Each stage of the pipeline stores intermediate outputs to facilitate debugging and validation.
- Unit Testing Readiness: Since data is available throughout the pipeline, individual modules can be refactored and tested in isolation before integration.
- Integration Testing: Running the full framework ensures all pipeline components work together as expected, mimicking large-scale operations.
- Scalability Testing: The framework is designed to handle an arbitrarily large feature set, constrained only by machine memory, ensuring future scalability.

### Getting Started
1. Copy the folder's config.env into the main repo directory so the /src modules can locate the test data and functions.
2. Complete or modify functions to test edge cases or walk through the framework.
3. Use the files named "test_<sub-dir name>" to test /src modules with the completed test configuration.