# Bayesian Zero-Shot Learning (BZSL) model

## Getting Started

This is the Python version of Bayesian classifier - BZSL.

The code included here is an adaptation of the original Python implementation of BZSL. Most of the instructions below
are preserved from the original code but not necessarily our recommendation for how to use it. Please refer to the root
folder README.md for more instructions on setup and execution.

## Prerequisites

The code was implemented in Python 3.7.10 and utilized just 2 packages listed below. The platform I used was Windows 10. 
```
numpy=1.19.2 
scipy=1.6.1 
```

## Installing

To run the code, You may create a conda environment (assuming you already have miniconda3 installed) by the following command on terminal:

```
conda create --name bzsl --file requirements.txt
```

If you already have  Python installed, you may just go ahead and manually install numpy and scipy without creating virtual environment by `pip install numpy=1.19.2`.

## Data

You may download the data from [Dropbox](https://www.dropbox.com/sh/gt6tkech0nvftk5/AADOUJc_Bty3sqOsqWHxhmULa?dl=0). Please put dataset into `data` folder and move the `data` folder into the same directory which contains the folders for codes. Note that in att_splits.mat from `INSECT` folder there are 4 new items that are different than standard ZSL benchmark datasets. Please see the associated descriptions below:

* `bold_ids`: You can use this number to search that particular instance in BOLD database to get more info
* `ids`: Image name/id
* `species`: The scientific name of that specimen
* `nucleotides`: DNA barcode (COI gene sequence) of the specimen
* `nucleotides_aligned`: Aligned DNA sequences (see Section 3 in the paper for details)
* `att`: Attributes that are derived from DNA barcodes as explained in the paper
 
## Experiments

To run the code and reproduce results from the paper, first activate conda virtual environment (If you chose to use virtual environment) by
```
conda activate bzsl
```
Then navigate to the `BZSL-Python` folder and execute the folowing code:
```
# tune BZSL hyperparameters
python Demo.py --datapath path/to/dataset/folder --embeddings path/to/embeddings --tuning

# run specific hyperparameters
python Demo.py --datapath path/to/dataset/folder --embeddings path/to/embeddings --k0 0.1 --k1 10.0 -m 12500 -s 5.0 -K 1
```

Note that the path for the dataset folder should contain the res101.mat and att_splits.mat files for the input image 
features and split metadata, respectively.

Main options for input arguments  are listed below (for the detailed list please check the code):
```
datatset: INSECT, CUB
side_info: original, w2v, dna
```
