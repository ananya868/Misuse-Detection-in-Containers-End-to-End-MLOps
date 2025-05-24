# Misuse Detection in Containers - MLOps Project 

Misuse Detection in containers using classification algorithms, end-to-end implemented with best-suited MlOps tools and frameworks. From data ingestion, pre-processing to model building, evalutaion and deployment, the project demonstrates an overview of all the steps involved in a data science / ML project. 

```python
import joblib
model = joblib.load("tuning/artifacts/knn_v4.pkl") # best model - 96% acc

# Transformed Features  
features = [[-7.5, -7.03, 2.8, -0.5, 4.5, -1.4, -1.1, 2.07, -2.1]]

# Get prediction 
prediction = model.predict(features)
print(f"output - {prediction}") #

Output - [.3]
Node-RED RCE :
 Occurs when vulnerabilities in the Node-RED platform,
 such as misconfigured authentication, public exposure of the admin interface,
 or outdated versions, allow attackers to execute arbitrary commands.
 This can lead to unauthorized access, data theft, service disruption,
 or escalation of privileges within containerized or Kubernetes environments."""
```


### **Problem Statement:** 📄
Misuse detection is essential in kubernetes clusters or Docker based services. Here's why it is needed: 
- **Protect against known threats**: Misuse detection focuses on identifying specific, known attack patterns, to safeguard from vulnerabilities that have been previously exploited.
- **Security in Dynamic environments**: In containerized systems like Kubernetes, these are highly dynamic and often involve rapid scaling, updates. This can lead to vulnerabilities.
- **Enhanced System Resilience**: By preventing misuse, the service can have reduced downtime, protecting sensitive data, and maintaining trust in infrastructure.

**Dataset**: [Dataset link](https://www.kaggle.com/datasets/yigitsever/misuse-detection-in-containers-dataset)
> Collection of network flows, collected from a kubernetes cluster running microservices-based software. It has 10 different attack scenerios and benign network flows. The labels are Benign (0), CVE-2020-13379 (1), Node-RED Reconnaissance (2), Node-RED RCE (3), Node-RED Container Escape (4), CVE-2021-43798 (5), CVE‑2019‑20933 (6), CVE‑2021‑30465 (7), CVE‑2021‑25741 (8), CVE‑2022‑23648 (9), CVE‑2019‑5736 (10) and DSB Nuclei Scan (11). The collected network traffic during attack scenarios and benign usage have been converted into tabular network flows (this dataset) using the updated CICFlowMeter tool https://github.com/GintsEngelen/CICFlowMeter.

> The dataset contains 86 features and 1 target column, with total of For the current dataset read the AINA 2024 paper: https://link.springer.com/chapter/10.1007/978-3-031-57942-4_22

### **Project Details:**
The project has been divided into multiple parts, be it statistical analysis, data pipelines, model building, experiments/tuning, deployment setups, or maintaining modular/reusable code base. Lets break it one by one:

- **EDA**: Exploring the data at first hand is crucial step for understanding the data structure, analyzing underlying distribution, deriving correlations, handling anomolies in data, process and formulate trainable features, selecting the best features, etc. For detailed lookout, please refer to dir "EDA", contains all steps from data cleaning to extracting the best features, everything implemented in jupyter notebooks.
 
- **Machine Learning Pipeline**: Its a structured sequence of steps that automates the process of preparing data, training models, and deploying them, ensuring a repeatable, efficient, and streamlined workflow for building and maintaining ML systems. The source code for core features are implemented in 'src' dir, and then utilized in 'steps' dir. This ensures that the main code remains intact, and eases debugging. You can checkout a sample run of steps in 'notebooks/verify_workflow.ipynb'. Abstract classes and methods have been used to improve reusability and maintainability, design patterns to enable polymorphism, flexibility in object creation.

- **Experimentation and Hyperparameter tuninig**: Improves model accuracy, helping the model to generalize well on unseen data. GridSearch, Optuna has been used for searching the best params. Mlflow is utilized to track metrices and parameters and select the best model configuration. Refer to dir 'tuning', to see details on experimental runs and final results. 'tuning/sorted_by_acc.png' list of models tested with their scores.

- **Deployment Setup**: Helps transform models from research prototypes into operational tools that provide real-time prediction in production environment. Several tools have been utilized in this project to ensure eased deployment -
   - *DVC*: Data version control for tracking data changes and retaining data variations
   - *FastAPI*: Enables seamless integration, real-time predictions, centralized maintenance, etc.
   - *CI/CD via Github Actions*: Automates testing, model training, and monitoring
   - *Docker*: Consistent environment, ensuring reproducibility, and easing dependency management
   - *Feast*: Pulling real-time features for inference, or historical features for training
 
### **Development Stack** 📦

- **Data Versioning and storage** - dvc, minio 
- **API Framework** - FastAPI, Pydantic
- **CI/CD** - Github Actions 
- **Experiment Tracking** - mlflow 
- **Model Training** - Scikit-learn 
- **EDA** - Pandas, matplotlib, seaborn, stats 
- **Feature and Artifact Store** - Feast, minio 
- **Feature Selection** - Optuna (try) 
- **Hyperparameter Tuning** - Optuna, GridSearch 
- **logging** - loguru 
- **Model Registry** - mlflow 
- **Lang** - Python3.11.5
- **Testing** - PyTest 
- **Environment Management** - Poetry, venv, pip

## **How to Use?** 
There are several ways to use the model, simplest one being API call to get predictions. Follow the instructions given below to setup the repo in your local and get started with the model -

### How to install?
Use Python 3.11.0 - 3.11.8 to avoid environment packages conflict. To set it up in your local -

```bash
git clone https://github.com/ananya868/Misuse-Detection-in-Containers-End-to-End-MLOps.git
cd Misuse-Detection-in-Containers-End-to-End-MLOps.git
```
Create a virtual env, activate it and run -
```bash
pip install -r requirements.txt
```
Thats it! 

### Usage
Simple Python code to use best model from pickle file -
```python
import joblib
model = joblib.load("tuning/artifacts/knn_v4.pkl") # best model 
features = [[1.1, 2.1, 1.4, 2.2, 9.4, 1.7, 6.4, 7.0, 2.0]] # Sample, (1, 9)
prediction = model.predict(features)
print(f"output - {prediction}")
```

To get predictions in real-time through an API call, run the following cmd - 
```bash
uvicorn app:app --port 8000 --reload
```
This runs the application using uvicorn. Now, you can use test the endpoints through -
```python
import requests 
url = "http://127.0.0.1:8000/predict/"
data = {
   "features": [1.1, 2.1, 1.4, 2.2, 9.4, 1.7, 6.4, 7.0, 2.0]
}
response = requests.post(url, json=data)
print(response.json()) 
```
or using curl 

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"input_data": "value"}'
```

**Github Actions** to automate data pipeline, model building and evaluation. Loguru to create logs at each step to track errors and processes, refer to 'ml_workflow.log' to see log history. To run data pipelines, simply run the following code in the root dir - 
```python
from steps.data_ingestion_step import data_ingestion_step
from steps.data_cleaning_step import data_cleaning_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.feature_engineering_step import feature_engineering_step
from steps.feature_selection_step import feature_selection_step
from steps.data_split_step import data_split_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step

# ML Pipeline
data = data_ingestion_step("# Your data path") # refer to 'dataset' section above 
data = data_cleaning_step(data) 
data = data_preprocessing_step(data)
data = feature_engineering_step(data)
data = feature_selection_step(data) 
x_train, x_test, y_train, y_test = data_split_step(data)
# Using KNN (best model through experimentation)
pred, model = model_building_step(x_train, y_train, x_test, model_name = "K Nearest Neighbors Classifier")
score = model_evaluation_step(y_test, pred)
print(f"Best Score: {score}") # 96%
```
**Note** - In feature selection step, the undersampling and oversampling strategy are fixed and might not work with other variations of data. Please refer to src/feature_selection.py to find out more.

**Using Docker**: 

To use the model through docker, simply pull the docker image -
```bash
docker pull ananya8154/misuse-detection-model
```
Run the Docker container - 
```bash
docker run -d -p 8000:8000 ananya8154/misuse-detection-model
```
You can now access the service at http://localhost:8000 (or http://<server-ip>:8000 if on a remote server).

## **Directory Overview**
The file structure is spread into varying components. Given below is the complete layout and explaination of file structure in this repo. 

- .dvc (dvc config file and gitignore file)
- **EDA** (Exploratory Data Analysis)
  - data-inspection : contain script for data inspection
  - data-preprocessing : contain all pre-processing steps
  - descriptive-statistics (used to summarize and analyze datasets)
    - distribution-analysis : dist analysis using histograms, kde, box plots
    - summary-statistics : scripts for central tendency, dispersian analysis
  - feature-analysis (correlation, using scatter plot, heatmaps, etc)
    - bivariate-analysis
    - multivariate-analysis
  - models : Model training on prepared data
- **feature_store** (Feast Feature store)
  - feature_repo (main repo for feature store)
    - feature_def.py : script for featureview, entity definition
    - feature_store.yaml : Feast store configuration
    - data (contains features in offline store)
      - predictors.parquet : most relevant features
      - target.parquet : label
- mlartifacts (saved models, saved after mlflow runs)
- **notebooks**
- **pipelines** (development and deployment pipeline)
  - development_pipeline.py
  - deploymeny_pipeline.py
- **src** (implementation of core algorithms/methods)
  - data_ingestion.py
  - data_cleaning.py
  - data_preprocessing.py
  - feature_engineering.py
  - feature_selection.py
  - data_split.py
  - model_building.py
  - model_evaluation.py
- **steps** (steps to run ml pipeline)
  - data_ingestion_step.py
  - data_cleaning_step.py
  - data_preprocessing_step.py
  - feature_engineering_step.py
  - feature_selection_step.py
  - data_split_step.py
  - model_building_step.py
  - model_evaluation_step.py
- **tests** (Pytest test files)
- **tuning** (Experimental runs and hyperparameter tuning)
  - mlruns (mlflow runs folder)
  - feast_feature_store_prep.ipynb (preparing data to upsert in feast)
  - mlflow_experiments.ipynb (mlflow trials)
  - optuna_tuning.ipynb (Feature selection using optuna)
  - scores.parquet (scores data from 10+ models, best - knn_v4)
  - sorted_by_acc.png (best model by accuracy)
- .dvcignore (dvc ignore file)
- **Dockerfile** (Docker definition file)
- **app.py** (model as a FastAPI)
- data_v.dvc (dvc file for data versions)
- directory_template.py (a python script to create project dir)
- logging_config.py (loguru logging configuration)
- requirements.txt (dependencies)
- **ml_workflow.log** (log files)
- run_pipeline.py (run dev and deployment pipeline)

## **Contact**
Feel free to reach me out for collaborations, queries or anything! Also, if you liked the repo, please give a start to this repo! 

- Email: ananya8154@gmail.com
- linkedIn: https://www.linkedin.com/in/ananya8154/








 

