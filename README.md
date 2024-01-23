# Official Implementation for BatLiNet

This repository provide the complete code for reproducing the results of paper "Accurate battery lifetime prediction across diverse aging conditions with deep learning".

**_NOTE:_** To build features for `MIX-100`, you will need at least 128 GB memory. Also, to achieve efficiency, you need at least one GPU device for feature smoothing and learning algorithm.

## Environment Setup
### Docker

<!-- We recommend using the image [fingertap/nmi_reproduction](#) that contains all the necessary libraries for reproduction. Please refer to the [dockerfile](Dockerfile) for details. -->
We recommend using our Docker image to reproduce our work. It includes all preprocessed data, code, and environment variables. You can pull the Docker image to your local environment using the following command:

```bash
docker pull batteryml/batlinet:latest
```

After pulling the Docker image and running the Docker container, you can execute the scripts mentioned in the [Experiment Reproduction](#experiment-reproduction) section to reproduce our experiments.

### Conda
To start from a fresh installation of Ubuntu with GPU, first install [conda](#) and create a conda environment:

```bash
conda create -n nmi_reproduction
conda activate nmi_reproduction
```

Install PyTorch following the [official instructions](#) and then install the rest dependencies:

```bash
pip install -r requirements.txt
```

Install Microsoft font for figures:

```bash
$ apt install ttf-mscorefonts-installer && fc-cache -f
Arial.ttf: "Arial" "Regular"
```

## Dataset preparation

In our experiments, we used the following public datasets:

- MATR-1
- MATR-2
- SNL
- UL_PUR
- CLO
- HNEI
- RWTH
- CALCE

Simply run the following script to download the raw data files and preprocess them:

```bash
export PYTHONPATH=.:$PYTHONPATH
python scripts/download.py --output-path ./data/raw
python scripts/preprocess.py --input-path ./data/raw --output-path ./data/processed
```

## Experiment Reproduction

We run all our experiments by providing a config file to our CLI tool. You can find all the config files [here](configs/). To accelerate training and evaluation, as well as reducing the memory footprint, you can first build the cache for all configs:

```bash
./scripts/build_cache.sh
```

This will load the datasets and build features for different configs with different train-test split, feature definition, etc.

To run a single config file, substitude the file path to the config file you would like to run and execute the following command:

```bash
PYTHONPATH=. python scripts/pipeline.py YOUR_CONFIG_FILE --train True --evaluate True
```

To run a config with $n$ seeds starting from 0, run the following command:

```bash
./scripts/run_pipeline_with_n_seeds.sh YOUR_CONFIG_FILE NUMBER_OF_SEEDS
```

To execute all BatLiNet experiments across all datasets using 8 seeds, please utilize the command provided below:

```bash
./scripts/run_batlinet.sh
```


To run all experiments sequentially (for a comprehensive reproduction of our experiments including comparisons with other models), use the following command:

```bash
./scripts/run_all_configs.sh
```

## Figures and Tables

<!-- We provide the code for plotting the figures and tables in [fig&tab.ipynb](fig&tab.ipynb). -->
For a more comprehensive understanding of our work, we have provided Jupyter notebooks that are instrumental in creating nearly all of the tables and figures presented in our research paper. 

These notebooks are conveniently located in the [notebooks](notebooks) directory of our repository. We encourage you to explore these resources to gain deeper insights into our methodologies and results.

## Citation

If you find our code or algorithm useful in your research, please cite:

```bibtext
@article{zhang2023accurate,
  title={Accurate battery lifetime prediction across diverse aging conditions with deep learning},
  author={Zhang, Han and Li, Yuqi and Zheng, Shun and Lu, Ziheng and Gui, Xiaofan and Xu, Wei and Bian, Jiang},
  journal={arXiv e-prints},
  pages={arXiv--2310},
  year={2023}
}
```