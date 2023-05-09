# Singapore Housing Price Prediction

This repository maintain codes for generating prediction/insights by using data science techniques for housing market of Singapore

## Getting Started

Make sure Git is installed in your system.

To clone the repository:

```bash
git clone https://github.com/sakimilo/datameka_sg_house_prediction.git
```

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.8. It does not assume a particular version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Under root directory of this project, you should see below folder structure,
```
homeprice_submission
├── data
├── duckdb
├── outputs
├── sql
├── ura_data
├── .gitignore
├── environment.yml
├── ingestion.py
├── modelling.py
└── Readme.md
```

At the root directory, run below command in Bash, to set up conda environment,
```bash
conda env create -f environment.yml
```

Once the environment has been successfully set up, activate environment using below command line. 
Activate it differently for different shells or environment (a different command in linux environment)
```
conda activate house_prediction
```

## Run the program

Before running the python program, make sure you have necessary data placed at the right directory.
For `data/` and `ura_data/` folder, please download necessary data artifacts from 
https://drive.google.com/drive/folders/1TLyOXkAb__qXhgNOx5pCG4jUXNIzBtLV?usp=sharing

```
homeprice_submission
├── data
|     ├── cpi.csv
|     ├── train.csv
|     ├── test.csv
|     ├── ...
|     └── vacant.csv
├── duckdb
├── outputs
├── sql
|     └── ingestion.sql
├── ura_data
|     ├── ResidentialTransaction_1995_Q1.csv
|     ├── ResidentialTransaction_1995_Q2.csv
|     ├── ...
|     └── ResidentialTransaction_2022_Q3.csv
├── .gitignore
├── environment.yml
├── ingestion.py
├── modelling.py
└── Readme.md
```

Execute python file for data ingestion,
```bash
python ingestion.py
```

Execute python file for model training and prediction,
```bash
python modelling.py
```

## Built For

 - Python 3.8
