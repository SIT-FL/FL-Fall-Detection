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
    - Preliminary Exploratory Data Analysis
    - Feature Engineering
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

### Federated Learning
Relevant flags:
```
--exp_name EXP_NAME   name of the experiment
--seed SEED           global random seed
--device DEVICE       device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`
--data_path DATA_PATH
                    path to save & read raw data
--log_path LOG_PATH   path to save logs
--result_path RESULT_PATH
                    path to save results
--use_tb              use TensorBoard for log tracking (if passed)
--tb_port TB_PORT     TensorBoard port number (valid only if `use_tb`)
--tb_host TB_HOST     TensorBoard host address (valid only if `use_tb`)

--dataset DATASET     name of dataset to use for an experiment (NOTE: case sensitive)
                        - image classification datasets in `torchvision.datasets`,
                        - text classification datasets in `torchtext.datasets`,
                        - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
                        - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
                        
--test_size {Specificed Range: [-1.00, 1.00]}
                    a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)

--split_type {iid,unbalanced,patho,diri,pre}
                    type of data split scenario
                        - `iid`: statistically homogeneous setting,
                        - `unbalanced`: unbalanced in sample counts across clients,
                        - `pre`: pre-defined data split scenario

--model_name {TwoNN,TwoCNN,SimpleCNN,FEMNISTCNN,Sent140LSTM,LeNet,MobileNet,SqueezeNet,VGG9,VGG9BN,VGG11,VGG11BN,VGG13,VGG13BN,ResNet10,ResNet18,ResNet34,ShuffleNet,MobileNeXt,SqueezeNeXt,MobileViT,StackedLSTM,StackedTransformer,LogReg,M5,DistilBert,SqueezeBert,MobileBert}
                    a model to be used (NOTE: case sensitive)

--dropout {Specificed Range: [0.00, 1.00]}
                    dropout rate

--use_model_tokenizer
                    use a model-specific tokenizer (if passed)

--use_pt_model        use a pre-trained model weights for fine-tuning (if passed)

--num_layers NUM_LAYERS
                    number of layers in recurrent cells

--num_embeddings NUM_EMBEDDINGS
                    size of an embedding layer

--embedding_size EMBEDDING_SIZE
                    output dimension of an embedding layer

--init_type {normal,xavier,xavier_uniform,kaiming,orthogonal,truncnorm,none}
                    weight initialization method

--init_gain INIT_GAIN
                    magnitude of variance used for weight initialization

--algorithm {fedavg,fedsgd,fedprox,fedavgm}
                    federated learning algorithm to be used

--mu {Specificed Range: [0.00, 1000000.00]}
                    constant for proximity regularization term (valid only if the algorithm is `fedprox`)

--eval_type {local,global,both}
                    the evaluation type of a model trained from FL algorithm
                        - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client's local evaluation set)
                        - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
                        - 'both': combination of `local` and `global` setting
                        
--eval_fraction {Specificed Range: [0.00, 1.00]}
                    fraction of randomly selected (unparticipated) clients for the evaluation (valid only if `eval_type` is `local` or `both`)

--eval_every EVAL_EVERY
                    frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)

--eval_metrics {acc1,acc5,auroc,auprc,youdenj,f1,precision,recall,seqacc,mse,mae,mape,rmse,r2,d2} [{acc1,acc5,auroc,auprc,youdenj,f1,precision,recall,seqacc,mse,mae,mape,rmse,r2,d2} ...]
                    metric(s) used for evaluation

--K K                 number of total cilents participating in federated training

--R R                 number of total federated learning rounds

--C {Specified Range: [0.00, 1.00]}
                    sampling fraction of clients per round (full participation when 0 is passed)

--E E                 number of local epochs

--B B                 local batch size (full-batch training when zero is passed)

--beta1 {Specificed Range: [0.00, 1.00]}
                    server momentum factor

--no_shuffle          do not shuffle data when training (if passed)

--optimizer OPTIMIZER
                    type of optimization method (NOTE: should be a sub-module of `torch.optim`, thus case-sensitive)

--max_grad_norm {Specificed Range: [0.00, inf]}
                    a constant required for gradient clipping

--weight_decay {Specificed Range: [0.00, 1.00]}
                    weight decay (L2 penalty)

--momentum {Specificed Range: [0.00, 1.00]}
                    momentum factor

--lr {Specificed Range: [0.00, 100.00]}
                    learning rate for local updates in each client

--lr_decay {Specificed Range: [0.00, 1.00]}
                    decay rate of learning rate

--lr_decay_step LR_DECAY_STEP
                    intervals of learning rate decay

--criterion CRITERION
                    objective function (NOTE: should be a submodule of `torch.nn`, thus case-sensitive)
```

### Acknowledgements
---
This project is built by the Singapore Institute of Technology and the University of Glasgow. It is developed by *Peter Febrianto Afandy*, under the guidance of *Prof. Ng Pai Chet*

For more information, kindly contact [*Prof. Ng Pai Chet*](mailto:paichet.ng@singaporetech.edu.sg).