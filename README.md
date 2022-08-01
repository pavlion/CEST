# Contrast-Enhanced Semi-supervised Text classification with few labels (CEST)

Authors: Austin Cheng-Yun Tsai, Sheng-Ya Lin, Li-Chen Fu

This repo contains the Pytorch code used to train the models used in our paper [Contrast-Enhanced Semi-supervised Text Classification with Few Labels
](https://ojs.aaai.org/index.php/AAAI/article/view/21391), presented at AAAI 2022.

If you find this work useful in your research, please cite:

```
@inproceedings{DBLP:conf/aaai/TsaiLF22,
  author    = {Austin Cheng{-}Yun Tsai and
               Sheng{-}Ya Lin and
               Li{-}Chen Fu},
  title     = {Contrast-Enhanced Semi-supervised Text Classification with Few Labels},
  booktitle = {Thirty-Sixth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2022, Thirty-Fourth Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2022, The Twelveth Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2022 Virtual Event, February 22
               - March 1, 2022},
  pages     = {11394--11402},
  publisher = {{AAAI} Press},
  year      = {2022},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/21391},
  timestamp = {Tue, 12 Jul 2022 14:14:21 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/TsaiLF22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## How to use the code
Our code is developed under `python=3.8`, `torch=1.8`, `CUDA=11.2`.

Notice that if you are using CUDA > 11.1, please install pytorch by
```
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
Otherwise, install it by 
```
pip3 install torch==1.8.1
```

Other dependency is detailed in `requirements.txt`. Please install them before running our code.

### 1. Preparing data for training
We pre-process the data into pickle files, which contain pandas DataFrames. In the DataFrame, each row represents a data instance, and there are two columns "text" and "labels", which stores the sentence and its corresponding label, respectively. Pre-processing code is in `ssl_data/process.py`. Note that the dataset is provided by `datasets`, huggingface's NLP dataset library.


### 2. Continued Pre-training on Task-specific Unlabeled Data (Optional)
Continuing pre-training on task-specific unlabeled data has been shown to be an effective mechanism (in UDA) to obtain a good base encoder to initialize the teacher model for CEST. 

Note that this step is optional, and one can still get a good results without this step. However, to reproduce the results in the paper, this step is necessary.


If you want to run pre-training yourself, please run `train_mlm.sh` for continue pre-training.
```
bash train_mlm.sh
```


### 3. Training CEST
Please set up configurations in `run.py`.
Specifically, some of the configurations are:

#### - coef_contrastive: The coefficient for regularization term R.
#### - contrastive_type: The method for constructing sub-graph. (CEST: Reliable, which means we consider reliability for construct sub-graph.)
#### - sampling_scheme: The method for sample selection. (CEST: IG_sampling, which means we select the data by sampling with weight)
#### - loss_opt: The name of loss function used in training. (CEST: PHuberCrossEntropy, which is one of the loss-robust functions.)
#### - tau: degree of data without noise (CEST: tau=10 for PHuberCrossEntropy)
#### - transformers_model_name: path to the (continued) pre-trained model
#### - emb_dim: the dimension for the embedding space (CEST: 128)

After setting up the hyper-parameters, please run the code by

```
python src/run.py
```
