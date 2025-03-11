# fx_classify template v.0.1
## Workflow
*Prerequisite*
Update 'config.env' and copy into '/workspace' so your
data and module paths are correct.
### data_preparation
*Check the config.py file in '/data_preparation' before any 'src' module execution.*
1. Load data into a raw database using 'create_raw_database.py'.
2. Create feature calculating function and storage map in 'feature_engineering.py'.
3. Create target calculating function and storage map in 'target_engineering.py'.
4. Orchestrate the feature and target creation and storage in 'prepare_data.py'.
### training
*Check the config.py file in '/data_preparation' before any 'src' module execution.*
1. Inspect data to determine if cleaning is needed.
2. If cleaning is needed, create filter functions in 'process_data.py' to return primary key sets.
3. Align all project tables using collected bad keys.
4. Optionally modify features or targets with .engineer() and your custom functions in 'process_data.py'.
5. Optionally fit feature scalers and save .pkl files.
6. Save data in .tfrecord files.
7. Optionally inspect .tfrecord files.
### evaluation
*Check the config.py file in '/data_preparation' before any 'src' module execution.*
1. Save your specified model predictions in a database.
2. Run LIME, print metrics, plot confidence calibration, and/or filter outputs for production readiness evaluation.