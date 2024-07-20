# percepteron
 ML Model using percepteron


# Perceptron Implementation

This repository contains an implementation of the Perceptron algorithm, a fundamental machine learning algorithm used for binary classification tasks.

## Overview

The Perceptron algorithm is one of the simplest types of artificial neural networks and serves as the building block for more complex neural networks. This project demonstrates how to implement and train a Perceptron to perform binary classification. The implementation includes steps to preprocess the data, train the Perceptron, and evaluate its performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Details](#details)
- [Examples](#examples)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the notebook and reproduce the results, you need to have Python and the necessary libraries installed. Follow the steps below to set up your environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/samipsinghal/percepteron.git
    cd perceptron
    ```

2. Install the required packages using `pip`:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

To use this repository, open the `perceptron.ipynb` notebook in Jupyter Notebook or Jupyter Lab. You can run the cells in the notebook to understand the steps involved in the implementation of the Perceptron algorithm and see the results of various experiments.

## Project Structure

The repository is structured as follows:

- `perceptron.ipynb`: Main Jupyter Notebook containing the project implementation.
- `data/`: Directory containing the datasets used for training and testing.

## Details

### Data Preparation

The data used in this project is a simple binary classification dataset. It is loaded and preprocessed to be used for training the Perceptron.

```python
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.loadtxt('data/data.csv', delimiter=',')

# Split into features and labels
X = data[:, :-1]
y = data[:, -1]

# Normalize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)
