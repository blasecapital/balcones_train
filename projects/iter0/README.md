# iter0
## Overview
This project uses 48-hours of standardized hourly exchange rate OHLC's as features and sets SL and TP based on volatility but maintains a 2:1 SL-to-TP width. The goal is to create a baseline project which assists in the model training framework development while providing a foundation to build upon. This project will fully document each step through data preparation, data cleaning, model training, model evaluation, and model explainability. 

## Installation
Utilize each step's .yml file to create the appropriate environment. Training, evaluation, and explainability will require tensorflow-gpu for fast inference.

## Prerequisites
1. Clone the repo
2. Configure environments
3. Set pythonpath to /src
4. Configure each step's .config and the .env file in the main directory

## Usage 
The source modules store routine operations and flexibly accept each step's config and .env file entries which store the iteration's arguments. Customize the project's features and targets using custom scripts but ensure they include a "functions(df)" function to store all of the feature calculating functions and a "storage_map" dictionary to specify the save table and related headers. 

### Notes
IMPORTANT:
The hourly features include the current rows' timestamp OHLC data. Targets must be calculated beginning at idx+1 with the idx+1's adjusted open price. So, features at timestamp 01/01/2025 00:00:00 should begin target calculation using 01/01/2025 01:00:00's open price and following OHLC data but then saved at 01/01/2025 00:00:00. This enables easy feature and target loading.