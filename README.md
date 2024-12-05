# Misuse Detection in Containers - MlOps Project 

Misuse Detection in containers using classification algorithms, end-to-end implemented with best-suited MlOps tools and frameworks. From data ingestion, pre-processing to model building, evalutaion and deployment, the project demonstrates an overview of all the steps involved in a data science / ML project. 

### **Problem Statement:**
Misuse detection is essential in kubernetes clusters or Docker based services. Here's why it is needed: 
- **Protect against known threats**: Misuse detection focuses on identifying specific, known attack patterns, to safeguard from vulnerabilities that have been previously exploited.
- **Security in Dynamic environments**: In containerized systems like Kubernetes, these are highly dynamic and often involve rapid scaling, updates. This can lead to vulnerabilities.
- **Enhanced System Resilience**: By preventing misuse, the service can have reduced downtime, protecting sensitive data, and maintaining trust in infrastructure.

**Dataset**:
> Collection of network flows, collected from a kubernetes cluster running microservices-based software. It has 10 different attack scenerios and benign network flows. The labels are Benign (0), CVE-2020-13379 (1), Node-RED Reconnaissance (2), Node-RED RCE (3), Node-RED Container Escape (4), CVE-2021-43798 (5), CVE‑2019‑20933 (6), CVE‑2021‑30465 (7), CVE‑2021‑25741 (8), CVE‑2022‑23648 (9), CVE‑2019‑5736 (10) and DSB Nuclei Scan (11). The collected network traffic during attack scenarios and benign usage have been converted into tabular network flows (this dataset) using the updated CICFlowMeter tool https://github.com/GintsEngelen/CICFlowMeter.

> The dataset contains 86 features and 1 target column, with total of For the current dataset read the AINA 2024 paper: https://link.springer.com/chapter/10.1007/978-3-031-57942-4_22

### **Project Details:**
The project has been divided into multiple parts, be it statistical analysis, data pipelines, model building, experiments/tuning, deployment setups, or maintaining modular/reusable code base. Lets break it one by one:

- **EDA**: Exploring the data at first hand is crucial step for understanding the data structure, analyzing underlying distribution, deriving correlations, handling anomolies in data, process and formulate trainable features, selecting the best features, etc. For detailed lookout, please refer to dir "EDA", contains all steps from data cleaning to extracting the best features, everything implemented in jupyter notebooks.
 
- **Machine Learning Pipeline**: Its a structured sequence of steps that automates the process of preparing data, training models, and deploying them, ensuring a repeatable, efficient, and streamlined workflow for building and maintaining ML systems. The source code for core features are implemented in 'src' dir, and then utilized in 'steps' dir. This ensures that the main code remains intact, and eases debugging. You can checkout a sample run of steps in 'notebooks/verify_workflow.ipynb'. Abstract classes and methods have been used to improve reusability and maintainability, design patterns to enable polymorphism, flexibility in object creation.

- **Experimentation and Hyperparameter tuninig**: 




