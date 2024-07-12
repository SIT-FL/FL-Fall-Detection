### Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Repository Structure](#repository-structure)
5. [Program Usage](#program-usage)
6. [Acknowledgements](#acknowledgements)

### Overview
---
This project is developed to demonstrate the efficacy of using Federated Learning for the detection of falls.

It is part of the submissions for the workshop paper submissions of the [2024 Multimodal Human Behaviour Analysis with Federated Learning, collocated with IEEE World Forum on the Internet of Things](https://mhba-fl.github.io/).

### Dataset
---
The fall detection dataset is derived from the [*Heart Rate and IMU Sensor Data for Fall Detection*](https://github.com/nhoyh/hr_imu_falldetection_dataset) dataset published by Nho, Lim & Kwon ([2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970371)). 

The dataset captures 16 unique scenarios (classifiable under `fall` and `non-fall`) and sensor readings, as denoted below:

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

### Methodology
---
The following methodology was employed for analysis of fall detection:
1. Data Pre-processing and Preparation 
    - Dataset Creation
    - Preliminary Exploratory Data Analysis
    - Feature Engineering
        - Outlier Management
        - Feature Creation
        - Feature Reduction
        - Feature Analysis
        - Feature Selection
        - Dataset Subsetting
2. Data Mining
    - Initial Data Mining
    -  Hyper-parameter Fine-tuning
3. Analysis
    - Model Performance Comparison and Analysis

### Repository Structure
---
```
requirements.txt

README.md (this file)
```

### Program Usage
---
1. Create a Python `virtualenv` on your local environment:
    ```
    python3 -m venv .venv
    ```
2. Install the necessary project dependencies:
    ```
    pip3 install -r requirements.txt
    ```
3. Run the interactive Python notebook to train/test the model, ensuring that you've linked the notebook to the correct Python `virtualenv`. 

### Acknowledgements
---
This project is built by the Singapore Institute of Technology and the University of Glasgow. It is developed by *Peter Febrianto Afandy*, under the guidance of *Prof. Ng Pai Chet*

For more information, kindly contact [*Prof. Ng Pai Chet*](mailto:paichet.ng@singaporetech.edu.sg).