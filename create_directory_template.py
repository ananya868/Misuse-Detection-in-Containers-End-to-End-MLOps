import os 
from pathlib import Path 

name = 'scripts'


# List of files to be created in the directory
list_of_files = [ # update this list to add new files
    # data
    f"{name}/data/__init__.py",  
    # src
    f"{name}/src/__init__.py",
    f"{name}/src/data_ingestion.py",
    f"{name}/src/data_cleaning.py",
    f"{name}/src/data_preprocessing.py",
    f"{name}/src/feature_engineering.py", 
    f"{name}/src/feature_selection.py",
    f"{name}/src/data_split.py",
    f"{name}/src/model_building.py", 
    f"{name}/src/model_evaluation.py",
    # steps
    f"{name}/steps/__init__.py",
    f"{name}/steps/data_ingestion_step.py",
    f"{name}/steps/handle_missing_value_step.py",
    f"{name}/steps/handle_outliers_step.py",
    f"{name}/steps/scaling_and_normalization_step.py",
    f"{name}/steps/handle_imbalance_in_data_step.py",
    f"{name}/steps/dimensionality_reduction_step.py",
    f"{name}/steps/model_building_step.py",
    f"{name}/steps/model_evaluation_step.py",
    # pipelines
    f"{name}/pipelines/__init__.py",
    f"{name}/pipelines/data_pipeline.py",
    f"{name}/pipelines/model_pipeline.py",
    # might create deployment pipeline later

    # tests 
    f"{name}/tests/__init__.py",
    
    # files 
    f"config.yaml",
    f"{name}/run_pipelines.py",

    "app.py", 
    "requirements.txt", 
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
]

# Function to create template 
def create_template(file_list, main_dir):
    """
    Create a directory template with the given list of files.
    """
    for f in file_list: 
        filepath = Path(f)
        file_dir, filename = os.path.split(filepath)

        if file_dir != "":
            os.makedirs(f"{file_dir}", exist_ok=True)
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass 
        else:
            print(f"File is already present at: {filepath}")

    print(f"--directory-updated--")


# Run the function
create_template(list_of_files, name)  