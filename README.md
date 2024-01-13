# Official Implementation for BatLiNet

This repository provide the complete code for reproducing the results of paper "Accurate battery lifetime prediction across diverse aging conditions with deep learning".

## Environment Setup

We recommend using the image [fingertap/nmi_reproduction](#) that contains all the necessary libraries for reproduction. Please refer to the [dockerfile](Dockerfile) for details.

To start from a fresh installation of Ubuntu with GPU, first install [conda](#) and create a conda environment:

```bash
conda create -n nmi_reproduction
conda activate nmi_reproduction
```

Install PyTorch following the [official instructions](#) and then install the rest dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

In our experiments, we used the following public datasets:

- MATR

Simply run the following script to download the raw data files and preprocess them:

```bash
export PYTHONPATH=.:$PYTHONPATH
python scripts/download.py --output-path ./data/raw
python scripts/preprocess.py --input-path ./data/raw --output-path ./data/processed
```

## Experiment Reproduction

We run all our experiments by providing a config file to our CLI tool. You can find all the config files [here](configs/). To run a single config file, substitude the file path to the config file you would like to run and execute the following command:

```bash
```

To run all experirments sequentially, use the following command:

```bash
```

## Figures and Tables

We provide the code for plotting the figures and tables in [fig&tab.ipynb](fig&tab.ipynb).

## Citation

If you find our code or algorithm useful in your research, please cite:

```bibtext
```