# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is developed to predict customer churn using bank data. 



## Files and data description
The data used in this project must be stored in data directory.
The data type must be csv file with `bank_data.csv` file name.
You can access data used in this project from Kaggle website.
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code
This repository is organized as below:
```
.
├── README.md
├── data
│   └── bank_data.csv
├── images
│   ├── classification_report_lr.png
│   ├── classification_report_rf.png
│   ├── correlations.png
│   ├── feature_importances_rf.png
│   ├── histograms.png
│   ├── roc_curve.png
│   └── value_counts.png
├── logs
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── notebooks
│   ├── Guide.ipynb
│   └── churn_notebook.ipynb
├── parameters.yml
├── requirements_py3.8.txt
├── sequencediagram.jpeg
├── src
│   └── churn_library.py
└── test
    └── churn_script_logging_and_tests.py
```
## Environment
This project assume python `version 3.8`.
You can build environment as following steps using `conda`.
 
 - `conda create -n predict_customer_churn python=3.8`
 - `conda activate predict_customer_churn`
 - `pip install -r requirements_py3.8.txt`

## Running Files
Before you run, please check if the code work correctly. To do this, run `pytest` as following steps.

- Open terminal
- Move directory to the project root directory
- Run `conda activate predict_customer_churn`
- Run `pytest`

You must pass all the test implemented in test/test/test_churn_script_logging.py
After checking all test has passed, run 
`python src/churn_library.py` to build model.
After Finished running, Logistic Regression model and Random Forest Classifier model are saved in `models` directory.
You can verify performances of each models by checking classification report, feature_importances, roc_curve stored in `images` directory.