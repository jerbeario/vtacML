# vtacML

vtacML is a machine learning package designed for the analysis of data from the Visible Telescope (VT) on the SVOM mission. This package uses machine learning models to analyze a dataframe of features from VT observations and identify potential gamma-ray burst (GRB) candidates. The primary goal of vtacML is to integrate into the SVOM data analysis pipeline and add a feature to each observation indicating the probability that it is a GRB candidate.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Grid Search and Model Training](#grid-search-and-model-training)
  - [Loading and Using the Best Model](#loading-and-using-the-best-model)
  - [Using Pre-trained Model for Immediate Prediction](#using-pre-trained-model-for-immediate-prediction)
  - [Config File](#config-file)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

## Overview

The SVOM mission, a collaboration between the China National Space Administration (CNSA) and the French space agency CNES, aims to study gamma-ray bursts (GRBs), the most energetic explosions in the universe. The Visible Telescope (VT) on SVOM plays a critical role in observing these events in the optical wavelength range.

vtacML leverages machine learning to analyze VT data, providing a probability score for each observation to indicate its likelihood of being a GRB candidate. The package includes tools for data preprocessing, model training, evaluation, and visualization.

## Installation

To install vtacML, you can use `pip`:

```sh
pip install vtacML
```

Alternatively, you can clone the repository and install the package locally:

```sh
git clone https://github.com/jerbeario/vtacML.git
cd vtacML
pip install .
```

## Usage

### Quick Start

Here’s a quick example to get you started with vtacML:

```python
from vtacML.pipeline import VTACMLPipe

# Initialize the pipeline
pipeline = VTACMLPipe()

# Load configuration
pipeline.load_config('path/to/config.yaml')

# Train the model
pipeline.train()

# Evaluate the model
pipeline.evaluate('evaluation_name', plot=True)

# Predict GRB candidates
predictions = pipeline.predict(observation_dataframe, prob=True)
print(predictions)
```

### Grid Search and Model Training

vtacML can perform grid search on a large array of models and parameters specified in the configuration file. Initialize the `VTACMLPipe` class with a specified config file (or use the default) and train it. Then, you can save the best model for future use.

```python
from vtacML.pipeline import VTACMLPipe

# Initialize the pipeline with a configuration file
pipeline = VTACMLPipe(config_file='path/to/config.yaml')

# Train the model with grid search
pipeline.train()

# Save the best model
pipeline.save_best_model('path/to/save/best_model.pkl')
```

### Loading and Using the Best Model

After training and saving the best model, you can create a new instance of the `VTACMLPipe` class and load the best model for further use.

```python
from vtacML.pipeline import VTACMLPipe

# Initialize a new pipeline instance
pipeline = VTACMLPipe()

# Load the best model
pipeline.load_best_model('path/to/save/best_model.pkl')

# Predict GRB candidates
predictions = pipeline.predict(observation_dataframe, prob=True)
print(predictions)
```

### Using Pre-trained Model for Immediate Prediction

If you already have a trained model, you can use the quick wrapper function `predict_from_best_pipeline` to predict data immediately. A pre-trained model is available by default.

```python
from vtacML.pipeline import predict_from_best_pipeline

# Predict GRB candidates using the pre-trained model
predictions = predict_from_best_pipeline(observation_dataframe, model_path='path/to/pretrained_model.pkl')
print(predictions)
```

### Config File

The config file is used to configure the model searching process. 

```yaml
# Default config file, used to search for best model using only first two sequences (X0, X1) from the VT pipeline
Inputs:
  file: 'combined_qpo_vt_all_cases_with_GRB_with_flags.parquet' # Data file used for training. Located in /data/
#  path: 'combined_qpo_vt_with_GRB.parquet'
#  path: 'combined_qpo_vt_faint_case_with_GRB_with_flags.parquet'
  columns: [
    "MAGCAL_R0",
    "MAGCAL_B0",
    "MAGERR_R0",
    "MAGERR_B0",
    "MAGCAL_R1",
    "MAGCAL_B1",
    "MAGERR_R1",
    "MAGERR_B1",
    "MAGVAR_R1",
    "MAGVAR_B1",
    'EFLAG_R0',
    'EFLAG_R1',
    'EFLAG_B0',
    'EFLAG_B1',
    "NEW_SRC",
    "DMAG_CAT"
    ] # features used for training
  target_column: 'IS_GRB' # feature column that holds the class information to be predicted

# Set of models and parameters to perform GridSearchCV over
Models:
  rfc:
    class: RandomForestClassifier()
    param_grid:
      'rfc__n_estimators': [100, 200, 300]  # Number of trees in the forest
      'rfc__max_depth': [4, 6, 8]  # Maximum depth of the tree
      'rfc__min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
      'rfc__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
      'rfc__bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
  ada:
    class: AdaBoostClassifier()
    param_grid:
      'ada__n_estimators': [50, 100, 200]  # Number of weak learners
      'ada__learning_rate': [0.01, 0.1, 1]  # Learning rate
      'ada__algorithm': ['SAMME']  # Algorithm for boosting
  svc:
    class: SVC()
    param_grid:
      'svc__C': [0.1, 1, 10, 100]  # Regularization parameter
      'svc__kernel': ['poly', 'rbf', 'sigmoid']  # Kernel type to be used in the algorithm
      'svc__gamma': ['scale', 'auto']  # Kernel coefficient
      'svc__degree': [3, 4, 5]  # Degree of the polynomial kernel function (if `kernel` is 'poly')
  knn:
    class: KNeighborsClassifier()
    param_grid:
      'knn__n_neighbors': [3, 5, 7, 9]  # Number of neighbors to use
      'knn__weights': ['uniform', 'distance']  # Weight function used in prediction
      'knn__algorithm': ['ball_tree', 'kd_tree', 'brute']  # Algorithm used to compute the nearest neighbors
      'knn__p': [1, 2]  # Power parameter for the Minkowski metric
  lr:
    class: LogisticRegression()
    param_grid:
      'lr__penalty': ['l1', 'l2', 'elasticnet']  # Specify the norm of the penalty
      'lr__C': [0.01, 0.1, 1, 10]  # Inverse of regularization strength
      'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # Algorithm to use in the optimization problem
      'lr__max_iter': [100, 200, 300]  # Maximum number of iterations taken for the solvers to converge
  dt:
    class: DecisionTreeClassifier()
    param_grid:
      'dt__criterion': ['gini', 'entropy']  # The function to measure the quality of a split
      'dt__splitter': ['best', 'random']  # The strategy used to choose the split at each node
      'dt__max_depth': [4, 6, 8, 10]  # Maximum depth of the tree
      'dt__min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
      'dt__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node

# Output directories
Outputs:
  model_path: '/output/models'
  viz_path: '/output/visualizations/'
  plot_correlation:
    flag: True
    path: 'output/corr_plots/'


```

## Documentation

See documentation at 


### Setting Up Development Environment


To set up a development environment, you can use the provided `requirements-dev.txt`:


```sh

conda create --name vtacML-dev python=3.8

conda activate vtacML-dev

pip install -r requirements.txt

```


### Running Tests


To run tests, use the following command:


```sh

pytest

```


## License


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## Contact


For questions or support, please contact:


- Jeremy Palmerio - [palmerio.jeremy@gmail.com](mailto:palmerio.jeremy@gmail.com)

- Project Link: [https://github.com/jerbeario/vtacML](https://github.com/jerbeario/VTAC_ML)
