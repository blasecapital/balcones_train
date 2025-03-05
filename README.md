# balcones_train
## **balcones_train:** expand and standardize your machine learning

## What is it?
**balcones_train** is a TensorFlow and CUDA-based framework for training neural networks on a single, local machine while working with arbitrarily large datasets and enforcing documentation. It removes cloud dependency for small to medium-sized projects by chunking data at every stage—cleaning, preprocessing, training, and evaluation—allowing models to handle more data. You are only limited by your machine's capacity to store the full dataset in drive and a chunk of training data and model weights in RAM. The framework gives you custom control over feature and target engineering, model architecture, and business impact modeling functions for iterative development. Reproducibility is built in by storing custom functions, datasets, model architecture, and arguments, keeping models ready for production. 

Blase Capital Management, LLC (a quantitative, Forex trading firm) began work on 'balcones_train' in 2024.

## Table of Contents
- [Workflow & Demo](#workflow)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Project Status & Roadmap[(#project-status-&-roadmap)
- [Why 'balcones_train'](#why-balcones_train?)
- [License](#license)

## Workflow
Watch the [demo](https://youtu.be/XsPsbfBBHbo).

### 1. Set Up a New Project
- Copy the contents of `/test` to `/projects`.
- Create a new iteration directory (e.g., `/projects/project0`).
- Move `/data_preparation`, `/training`, and `/evaluation` into the new subdirectory.
- Customize `config.env` in the main directory to store `PYTHONPATH`, project configuration, and data file paths.

### 2. Prepare Data
- **Load Raw Data:** Create a custom script to load raw data into a SQLite database in `/data`.
- **Feature Engineering:** Modify the example feature and target engineering scripts to generate the training dataset.
- **Configuration:** Customize the `/data_preparation/config.py` file to specify how to load and filter source data.
- **Run Preparation:** Execute the `data_preparation` script.

### 3. Clean, Prepare, and Train
- **Set Training Modules:** In `training.py`, adjust function arguments to specify which modules to use (`CleanRawData`, `PrepData`, `Train`).
- **Configuration Check:** Update `/projects/<project>/training/config.py` at each step to ensure correct arguments.
- **Data Inspection:** Use `CleanRawData.inspect_data` to visualize and describe your data.
- **Filtering & Cleaning:**
  - Modify `process_raw_data.py` to define feature/target filtering criteria.
  - Run `CleanRawData.clean` if filtering data to collect `primary_keys` for removal.
  - Run `CleanRawData.align` to create a cleaned database.
- **Feature Engineering & Scaling:**
  - Adjust `process_raw_data.py` for feature/target modifications and execute `PrepData.engineer`.
  - Apply scaling using `PrepData.scale`, storing scalers for applicable database tables.
- **Dataset Storage:** Run `PrepData.save_batches` to store datasets as `.tfrecord` files and optionally validate them with `PrepData.validate_record`.
- **Model Training:**  
  - Customize `training_funcs.py` to define loss functions, callbacks, and model architecture.  
  - Train models using `Train.train_models()`.

### 4. Evaluate Performance
- **Configuration Check:** Update `/projects/<project>/evaluation/config.py` at each step.
- **Evaluation Steps:**  
  - Use `evaluation.py` to store predictions, run LIME on a single observation, report metrics such as accuracy, and visualize calibration.  
  - Assess if the model meets production requirements.  
- **Business Impact:** Modify `eval_funcs.py` to model the impact of predictions.

## Dependencies
1. **CUDA & cuDNN:**  
   - Install **CUDA 11.2** and **cuDNN 8.1** (latest compatible version for Windows).  
   - Follow this [installation guide](https://youtu.be/hHWkvEcDBO0?si=3xxz4VfhOVcnri8E).  
2. **Project Environment:**  
   - All dependencies are listed in **`balcones_train_env.yml`**, including:  
     - **TensorFlow-GPU 2.5.0**  
     - **Pandas 1.1.5**  
     - **NumPy 1.19.5**  
3. **Create the Conda Environment:**  
   ```sh
   conda env create -f balcones_train_env.yml
   conda activate balcones_train
   ```

## Getting Started
1. Clone repository
2. Setup the environment using balcones_train_env.yml and Anaconda Prompt
   ```sh
   conda env create -f balcones_train_env.yml
   conda activate balcones_train
   ```
3. Configure your IDE by setting the working directory to the project's main folder and activate the 'balcones_train' environment
4. Install/update NVIDIA driver
5. Run the following to verify installation:
   ```sh
   import tensorflow as tf

   print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())
   print("GPU device name:", tf.test.gpu_device_name())
   ```
6. Walk through the tests folder to become acquainted with the modules

## Project Status & Roadmap
### Current Version: v0.1.0 (Pre-Release)
This is an initial public version of `balcones_train`. The core framework is functional for a niche use case (training trade-signal generating models using timeseries and technical features using exchange rates as the primary data source), but further refinements and optimizations are planned to expand applicability.

### Future Development:
Version 1.0 will transform 'balcones_train' into a flexible framework for training neural networks across multiple machine learning tasks, while maintaining its core value proposition:
Enabling local training with arbitrarily large datasets for small to medium-sized projects, while standardizing the full ML pipeline (cleaning, preprocessing, training, evaluation) and enforcing documentation for production-ready models.

Planned enhancements:
- [ ] **Expand beyond classification:** Support regression tasks and reinforcement learning (computer vision and language modeling may be included or reserved for future versions).
- [ ] **Increase flexibility:** Refactor modules to handle various data types and projects outside of quant trading.
- [ ] **Expand tests/examples:** Demonstrate and verify versatility and make project setup easier.
- [ ] **Improve documentation:** Project should be easy to understand and contribute to.

## Why `balcones_train`?
Independent practitioners and small startups working with **datasets too large for memory** can streamline their workflow using `balcones_train`'s **batching functions**. Organizations handling **sensitive data** can improve security by training models **locally**. Users can also **prototype their models before committing to a cloud-based platform like Snowflake**.

Beyond cost and security benefits, `balcones_train` **builds in best practices** to prevent **data leakage** and **ensure reproducibility**, a [prevailing issue in machine learning](https://reproducible.cs.princeton.edu/).

### **Differentiators**
1. **Avoid Cloud Costs** – No expensive GPU instances or unpredictable billing.  
2. **Data Privacy & Security** – Keeps datasets offline, reducing compliance risks.  
3. **Full Control** – Optimized environment, faster debugging, and no cloud restrictions.  
4. **Faster Iteration Cycles** – Immediate hardware access with no cloud latency or interruptions.  
5. **Better Reproducibility** – Enforces structured documentation and standardized workflows.

### **Considerations**
- **Storage Constraints** – Each training iteration **stores processed data, model weights, training metadata, and predictions**, which can **quickly consume disk space**. Manual memory management may be required on machines with limited storage.  
- **Cloud Portability** – While `balcones_train` is designed for **local-first training**, it can be adapted to **Google Cloud, Colab, or other platforms** with **careful path configuration**. However, moving to the cloud may **compromise reproducibility** if best practices—such as requirement tracking and configuration logging—are not maintained.  

## License
This project is licensed under the BSD 3-Clause License.

### Third-Party Dependencies
This project relies on open-source dependencies specified in `balcones_train_env.yml`.  
Each dependency is subject to its own license, which can be found in their respective repositories.

