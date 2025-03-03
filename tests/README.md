# Test Framework for Model Training
## Overview
This repository serves as a template-based integration testing framework for validating machine learning model training workflows. It is designed to ensure that each step of the pipeline—from data preparation to training and evaluation—is functional and scalable, allowing for seamless development and debugging before deployment.

By using a controlled, dummy dataset, this framework simulates the expected training process at scale while providing a structured environment for unit testing, integration testing, and framework validation.

## Why This Approach?
This framework is built on the principle of template-based integration testing, meaning that:
- It mirrors real-world model training workflows but on a smaller, controlled dataset.
- It enables step-by-step validation of each pipeline component before scaling to large datasets.
- It supports both modular unit testing and full pipeline integration testing to detect issues early.
- It serves as a blueprint for future projects, ensuring reproducibility and consistency.
- It ensures data availability across steps, aiding debugging and iterative improvement.

## Folder Structure
```sh
/test 
│── /data_preparation  
│   ├── config.py
│   ├── test_create_dummy_data_db.py  
│   ├── test_feature_engineering.py  
│   ├── test_target_engineering.py
│   ├── test_data_preparation.py
│   └── README.md  
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
│   ├── test_training.py
│   └── README.md  
│  
│── /evaluation  
│   ├── /predictions
│   ├── config.py  
│   ├── test_custom_eval_funcs.py
│   ├── test_evaluation.py
│   └── README.md  
│  
│── /data
│
│── config.env
│ 
└── README.md  
```

## Detailed Module Breakdown
Data Preparation (/data_preparation)
Purpose: Stores raw data and applies feature engineering and target generation to prepare inputs for training.
Testing Focus: Ensures that features and targets are generated correctly and are aligned for model training.

## Training (/training)
Purpose: Handles data cleaning, model training, and hyperparameter tuning.
Testing Focus: Ensures that the model is trained using clean, correctly formatted data as expected.

## Evaluation (/evaluation)
Purpose: Loads trained models, generates predictions, and evaluates performance.
Testing Focus: Ensures that predictions are generated correctly, model explanations are valid, metrics are computed reliably, and model calibration is appropriate.

## How This Framework Supports Testing
- Step-by-Step Data Availability: Each stage of the pipeline stores intermediate outputs to facilitate debugging and validation.
- Unit Testing Readiness: Since data is available throughout the pipeline, individual modules can be refactored and tested in isolation before integration.
- Integration Testing: Running the full framework ensures all pipeline components work together as expected, mimicking large-scale operations.
- Scalability Testing: The framework is designed to handle an arbitrarily large feature set, constrained only by machine memory, ensuring future scalability.

## Getting Started
1. Copy the folder's config.env into the main repo directory so the /src modules can locate the test data and functions.
2. Complete or modify functions to test edge cases or walk through the framework.
3. Use the files named "test_<sub-dir name>" to test /src modules with the completed test configuration.