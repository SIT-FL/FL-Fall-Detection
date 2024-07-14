
# Federated Learning in PyTorch
Implementations of various Federated Learning (FL) algorithms in PyTorch, especially for research purposes.

## Implementation Details
### Datasets
* Supports all image classification datasets in `torchvision.datasets`.
* Supports all text classification datasets in `torchtext.datasets`.
* Supports all datasets in [LEAF benchmark](https://leaf.cmu.edu/) (*NO need to prepare raw data manually*)
* Supports additional image classification datasets ([`TinyImageNet`](https://www.kaggle.com/c/tiny-imagenet), [`CINIC10`](https://datashare.ed.ac.uk/handle/10283/3192)).
* Supports additional text classification datasets ([`BeerReviews`](https://snap.stanford.edu/data/web-BeerAdvocate.html)).
* Supports tabular datasets ([`Heart`, `Adult`, `Cover`](https://archive.ics.uci.edu/ml/index.php)).
* Supports temporal dataset ([`GLEAM`](http://www.skleinberg.org/data.html))
* __NOTE__: don't bother to search raw files of datasets; the dataset can automatically be downloaded to the designated path by just passing its name!
### Statistical Heterogeneity Simulations
* `IID` (i.e., statistical homogeneity)
* `Unbalanced` (i.e., sample counts heterogeneity)
* `Pathological Non-IID` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `Dirichlet distribution-based Non-IID` ([Hsu et al., 2019](https://arxiv.org/abs/1909.06335))
* `Pre-defined` (for datasets having natural semantic separation, including `LEAF` benchmark ([Caldas et al., 2018](https://arxiv.org/abs/1812.01097)))
### Models
* `LogReg` (logistic regression), `StackedTransformer` (TransformerEncoder-based classifier)
* `TwoNN`, `TwoCNN`, `SimpleCNN` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `FEMNISTCNN`, `Sent140LSTM` ([Caldas et al., 2018](https://arxiv.org/abs/1812.01097)))
* `LeNet` ([LeCun et al., 1998](https://ieeexplore.ieee.org/document/726791/)), `MobileNet` ([Howard et al., 2019](https://arxiv.org/abs/1905.02244)), `SqueezeNet` ([Iandola et al., 2016](https://arxiv.org/abs/1602.07360)), `VGG` ([Simonyan et al., 2014](https://arxiv.org/abs/1409.1556)), `ResNet` ([He et al., 2015](https://arxiv.org/abs/1512.03385))
* `MobileNeXt` ([Daquan et al., 2020](https://arxiv.org/abs/2007.02269)), `SqueezeNeXt` ([Gholami et al., 2016](https://arxiv.org/abs/1803.10615)), `MobileViT` ([Mehta et al., 2021](https://arxiv.org/abs/2110.02178))
* `DistilBERT` ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)), `SqueezeBERT` ([Iandola et al., 2020](https://arxiv.org/abs/2006.11316)), `MobileBERT` ([Sun et al., 2020](https://arxiv.org/abs/2004.02984))
* `M5` ([Dai et al., 2016](https://arxiv.org/abs/1610.00087))
### Algorithms
* `FedAvg` and `FedSGD` (McMahan et al., 2016) <a href='https://arxiv.org/abs/1602.05629'>Communication-Efficient Learning of Deep Networks from Decentralized Data</a>
* `FedAvgM` (Hsu et al., 2019) <a href='https://arxiv.org/abs/1909.06335'>Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification</a>
* `FedProx` (Li et al., 2018) <a href='https://arxiv.org/abs/1812.06127'>Federated Optimization in Heterogeneous Networks</a>
* `FedOpt` (`FedAdam`, `FedYogi`, `FedAdaGrad`) (Reddi et al., 2020) <a href='https://arxiv.org/abs/2003.00295'>Adaptive Federated Optimization</a>

### Evaluation schemes
* `local`: evaluate FL algorithm using holdout sets of (some/all) clients NOT participating in the current round. (i.e., evaluation of personalized federated learning setting)
* `global`: evaluate FL algorithm using global holdout set located at the server. (*ONLY available if the raw dataset supports pre-defined validation/test set*).
* `both`: evaluate FL algorithm using both `local` and `global` schemes.
### Metrics
* Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, F1
* Area under ROC, Area under PRC, Youden's J
* Seq2Seq Accuracy
* MSE, RMSE, MAE, MAPE
* $R^2$, $D^2$

## Requirements
* See `requirements.txt`. (I recommend building an independent environment for this project, using e.g., `Docker` or `conda`)
* When you install `torchtext`, please check the version compatibility with `torch`. (See [official repository](https://github.com/pytorch/text#installation))
* Plus, please install `torch`-related packages using one command provided by the official guide (See [official installation guide](https://pytorch.org/get-started/locally/)); e.g., `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge` 

## Configurations
* See `python3 main.py -h`.

```
usage: main.py [-h] --exp_name EXP_NAME [--seed SEED] [--device DEVICE] [--data_path DATA_PATH] [--log_path LOG_PATH]
               [--result_path RESULT_PATH] [--use_tb] [--tb_port TB_PORT] [--tb_host TB_HOST] --dataset DATASET
               [--test_size {Specificed Range: [-1.00, 1.00]}] [--rawsmpl {Specificed Range: [0.00, 1.00]}]
               [--resize RESIZE] [--crop CROP] [--imnorm] [--randrot RANDROT]
               [--randhf {Specificed Range: [0.00, 1.00]}] [--randvf {Specificed Range: [0.00, 1.00]}]
               [--randjit {Specificed Range: [0.00, 1.00]}] --split_type {iid,unbalanced,patho,diri,pre}
               [--mincls MINCLS] [--cncntrtn CNCNTRTN] --model_name
               {TwoNN,TwoCNN,SimpleCNN,FEMNISTCNN,Sent140LSTM,LeNet,MobileNet,SqueezeNet,VGG9,VGG9BN,VGG11,VGG11BN,VGG13,VGG13BN,ResNet10,ResNet18,ResNet34,ShuffleNet,MobileNeXt,SqueezeNeXt,MobileViT,StackedLSTM,StackedTransformer,LogReg,M5,DistilBert,SqueezeBert,MobileBert}
               [--hidden_size HIDDEN_SIZE] [--dropout {Specificed Range: [0.00, 1.00]}] [--use_model_tokenizer]
               [--use_pt_model] [--seq_len SEQ_LEN] [--num_layers NUM_LAYERS] [--num_embeddings NUM_EMBEDDINGS]
               [--embedding_size EMBEDDING_SIZE]
               [--init_type {normal,xavier,xavier_uniform,kaiming,orthogonal,truncnorm,none}] [--init_gain INIT_GAIN]
               --algorithm {fedavg,fedsgd,fedprox,fedavgm} --eval_type {local,global,both}
               [--eval_fraction {Specificed Range: [0.00, 1.00]}] [--eval_every EVAL_EVERY] --eval_metrics
               {acc1,acc5,auroc,auprc,youdenj,f1,precision,recall,seqacc,mse,mae,mape,rmse,r2,d2}
               [{acc1,acc5,auroc,auprc,youdenj,f1,precision,recall,seqacc,mse,mae,mape,rmse,r2,d2} ...] [--K K]
               [--R R] [--C {Specificed Range: [0.00, 1.00]}] [--E E] [--B B]
               [--beta1 {Specificed Range: [0.00, 1.00]}] [--no_shuffle] --optimizer OPTIMIZER
               [--max_grad_norm {Specificed Range: [0.00, inf]}] [--weight_decay {Specificed Range: [0.00, 1.00]}]
               [--momentum {Specificed Range: [0.00, 1.00]}] --lr {Specificed Range:
               [0.00, 100.00]} [--lr_decay {Specificed Range: [0.00, 1.00]}] [--lr_decay_step LR_DECAY_STEP]
               --criterion CRITERION [--mu {Specificed Range: [0.00, 1000000.00]}]

options:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   name of the experiment
  --seed SEED           global random seed
  --device DEVICE       device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`
  --data_path DATA_PATH
                        path to save & read raw data
  --log_path LOG_PATH   path to save logs
  --result_path RESULT_PATH
                        path to save results

    TensorBoard
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
  --rawsmpl {Specificed Range: [0.00, 1.00]}
                        a fraction of raw data to be used (valid only if one of `LEAF` datasets is used)

    Image
  --resize RESIZE       resize input images (using `torchvision.transforms.Resize`)
  --crop CROP           crop input images (using `torchvision.transforms.CenterCrop` (for evaluation) and `torchvision.transforms.RandomCrop` (for training))
  --imnorm              normalize channels with mean 0.5 and standard deviation 0.5 (using `torchvision.transforms.Normalize`, if passed)
  --randrot RANDROT     randomly rotate input (using `torchvision.transforms.RandomRotation`)
  --randhf {Specificed Range: [0.00, 1.00]}
                        randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)
  --randvf {Specificed Range: [0.00, 1.00]}
                        randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)
  --randjit {Specificed Range: [0.00, 1.00]}
                        randomly change the brightness and contrast (using `torchvision.transforms.ColorJitter`)

  --split_type {iid,unbalanced,patho,diri,pre}
                        type of data split scenario
                            - `iid`: statistically homogeneous setting,
                            - `unbalanced`: unbalanced in sample counts across clients,
                            - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
                            - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
                            - `pre`: pre-defined data split scenario
                            
  --mincls MINCLS       the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)
  --cncntrtn CNCNTRTN   a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)

  --model_name {TwoNN,TwoCNN,SimpleCNN,FEMNISTCNN,Sent140LSTM,LeNet,MobileNet,SqueezeNet,VGG9,VGG9BN,VGG11,VGG11BN,VGG13,VGG13BN,ResNet10,ResNet18,ResNet34,ShuffleNet,MobileNeXt,SqueezeNeXt,MobileViT,StackedLSTM,StackedTransformer,LogReg,M5,DistilBert,SqueezeBert,MobileBert}
                        a model to be used (NOTE: case sensitive)

  --hidden_size HIDDEN_SIZE
                        hidden channel size for vision models, or hidden dimension of language models
  --dropout {Specificed Range: [0.00, 1.00]}
                        dropout rate
  --use_model_tokenizer
                        use a model-specific tokenizer (if passed)
  --use_pt_model        use a pre-trained model weights for fine-tuning (if passed)
  --seq_len SEQ_LEN     maximum sequence length used for `torchtext.datasets`)
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
  --C {Specificed Range: [0.00, 1.00]}
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
  --mu {Specificed Range: [0.00, 1000000.00]}
                        constant for proximity regularization term (valid only if the algorithm is `fedprox`)
```

## Example Commands
* See shell files prepared in `commands` directory.

## TODO
- [ ] Support another model, especially lightweight ones for cross-device FL setting. (e.g., [`EdgeNeXt`](https://github.com/mmaaz60/EdgeNeXt))
- [ ] Support another structured dataset including temporal and tabular data, along with datasets suitable for cross-silo FL setting. (e.g., [`MedMNIST`](https://github.com/MedMNIST/MedMNIST))
- [ ] Add other popular FL algorithms including personalized FL algorithms (e.g., [`SuPerFed`](https://arxiv.org/abs/2109.07628)).
- [ ] Attach benchmark results of sample commands.

## Contact
Should you have any feedback, please create a thread in __issue__ tab. Thank you :)
