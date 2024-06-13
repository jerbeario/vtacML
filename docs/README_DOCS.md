
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
  - [Examples](#examples)
- [Documentation](#documentation)


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
git clone https://github.com/jerbeario/VTAC_ML.git
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

## Documentation
