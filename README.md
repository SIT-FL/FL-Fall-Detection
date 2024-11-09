# Federated Learning for Hierarchical Fall Detection and Human Activity Recognition

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Repository Structure](#repository-structure)
5. [Program Usage](#program-usage)
6. [Acknowledgements](#acknowledgements)

## Overview
This project provides the implementation for the paper, *Federated Learning for Hierarchical Fall Detection and Human Activity Recognition*, which  presents a federated learning framework for enhanced healthcare monitoring using a hierarchical two-stage approach for precise fall detection and human activity recognition.

The first stage involves binary classification for fall detection, to distinguish between fall and non-fall events. Subsequently, the second stage involves multi-class classification for precise human activity recognition, depending on the results of the first stage. If a fall is detected, the system classifies the type of fall to facilitate appropriate medical responses; if no fall is detected, it classifies the specific activity being performed. 

We evaluate each model's performance for the binary classification of fall and non-fall events, as well as for multi-class classification of time-series sensor readings, into 19 distinct scenarios, as prescribed by the dataset used. An LSTM model is used for both approaches, with standardised model hyperparameters.

It is part of the submissions for the workshop paper submissions of the [2024 Multimodal Human Behaviour Analysis with Federated Learning, collocated with IEEE World Forum on the Internet of Things](https://mhba-fl.github.io/).

## Dataset
The fall detection dataset is derived from the [*Heart Rate and IMU Sensor Data for Fall Detection*](https://github.com/nhoyh/hr_imu_falldetection_dataset) dataset published by Nho, Lim & Kwon ([2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970371)). 

The dataset captures 19 unique scenarios (classifiable under `fall` and `non-fall`) and sensor readings, as denoted below:

<ins>Scenarios</ins>
- `fall` class
    - `fall1`: Clockwise forward fall
    - `fall2`: Clocklwise backward fall
    - `fall3`: Right to left lateral fall
    - `fall4`: Counterclock-wise forward fall
    - `fall5`: Counterclock-wise backward fall
    - `fall6`: Left to right lateral fall
- `non-fall` class
    - `bed`: Lying down and up on the bed
    - `chair`: Sitting down and up
    - `clap`: Hitting the sensor
    - `cloth`: Wearing a cloth
    - `eat`: Eating
    - `hair`: Brushing the hair
    - `shoe`: Tying a shoelace
    - `stair`: Climbing up and down on stairs
    - `teeth`: Brushing teeth
    - `walk`: Walking
    - `wash`: Washing
    - `write`: Writing
    - `zip`: Rapidly zipping up and down

<ins>Sensor Readings</ins>
- `ax`: x axis of accelerometer signal (g)
- `ay`: y axis of accelerometer signal (g)
- `az`: z axis of accelerometer signal (g)
- `w`: quaternion of gyroscope
- `x`: quaternion of gyroscope
- `y`: quaternion of gyroscope
- `z`: quaternion of gyroscope
- `droll`: angular velocity of gyroscope
- `dpitch`: angular velocity of gyroscope
- `dyaw`: angular velocity of gyroscope
- `heart`: PPG sensor
- `time`: Real time 

## Methodology
The following methodology was employed for experimental analysis for fall detection classification:
1. Data Pre-processing and Preparation 
    - Preliminary Exploratory Data Analysis
    - Dataset Construction
2. Training of Comparative Model
3. Training of Federated Models (Local and Global)
3. Analysis
    - Model Performance Comparison and Analysis

## Repository Structure
```
/reference (reference source code that is unused)

/dataset (contains fall-detection dataset files, and associated Python notebooks)
    comparative_model.ipynb
    eda_preprocessing.ipynb
    config.py
    utils.py

/results (contains results of federated learning)

config.py (configuration file)

utils (utility functions)

run.py (main source code)

requirements.txt

README.md (this file)
```

## Program Usage
1. Create a Python `virtualenv` on your local environment:
    ```
    python3 -m venv .venv
    ```
2. Install the necessary project dependencies:
    ```
    pip3 install -r requirements.txt
    ```
3. To view or preprocess the dataset, run the [`eda.ipynb`](./dataset/eda.ipynb) notebook ensuring that you've linked the notebook to the correct Python `virtualenv`. 

4. To run the comparative model, run the [`comparative_model.ipynb`](./dataset/comparative_model.ipynb) notebook ensuring that you've linked the notebook to the correct Python `virtualenv`. 

5. To perform the federated learning experiment, run the following command:
    ```
    # For regular federated averaging
    python3 run.py

    # For FedProx
    python3 run.py --fedprox
    ```

## Acknowledgements
This project is built by the Singapore Institute of Technology and the University of Glasgow. It is developed by *Peter Febrianto Afandy*, under the guidance of *Prof. Ng Pai Chet*

For more information, kindly contact [*Prof. Ng Pai Chet*](mailto:paichet.ng@singaporetech.edu.sg).