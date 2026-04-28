
# Pick-a-pKa

To be modified!

# pKaLearn

Leveraging our Teaching Experience to Improve Machine Learning: Application to the Development of pKaLearn, a pKa Predictor.

Jérôme Genzling, Ziling Luo, Benjamin Weiser, Nicolas Moitessier

nicolas.moitessier@mcgill.ca

2023-12-07 – revised 2025-11-06

![Graphical Abstract](Graphical-abstract300.png)

# 🔍 What is pKaLearn?

A Graph Neural Network (GNN) model for:

- Predicting pKa values of ionizable centers (test, test_with_IC)
- Identifying protonation sites (infer)
- Estimating dominant protonation states at a given pH
- Supporting iterative protonation/deprotonation of polyprotic molecules

# 🧪 Core Functionalities

- Input: CSV with SMILES and (optionally) ionizable atom indices
- Output: pKa value(s), and major protonated species at given pH
- Iterative inference for molecules with multiple ionizable centers
- Easily extendable to new datasets or re-trainable on custom data

# 📦 Required Libraries

Install with pip:

pip install torch torch_geometric pandas numpy rdkit seaborn hyperopt

You can also recreate our virtual environment using environment.yml

# 📁 Repository Structure

Datasets/ : All cleaned, split, and raw datasets

Baseline_Models/Descriptors/ : Code to generate traditional descriptors

Baseline_Models/RF, /XGB : Traditional model training scripts (Random Forest/XGB)

GNN/ : All code related to GNN/GAT models

MolGpKa_retrained/ : Code and data for retraining MolGpKa

# 🚀 Getting Started with the GNN

## 0. Recommended file structure

To run pKaLearn, we recommend you to have in the same folder the following directories:

pKaLearn  
&emsp;|Datasets  
&emsp;&emsp;|pickled_data  
&emsp;|GNN  
&emsp;|Model  

You can then place your datasets in .csv format in the Datasets folder, and run the commands in the GNN folder.

## 1. See available options

python main.py --mode usage

All possible arguments and their default values will be printed.

## 2. Predict pKa on a sample set

Your CSV will need to have at least two columns: 'Name' and 'Smiles'

On Windows:

python main.py --mode infer --input your_input.csv > infer_your_input.out

On Linux: 

python main.py --mode infer --data_path ..\Datasets\ --input your_input.csv --infer_pickled ..\Datasets\pickled_data\infer_pickled.pkl --model_dir ..\Model\ > infer_your_input.out

## 3. Predict from a CSV in Python

If you want to use pKaLearn and test it against pKa values (experimental values, for instance), your values need to be in a "pKa" column

Then, run the following command:

python main.py --mode test --input your_input.csv > test_your_input.out

If you want to test your molecules (which must be in the acidic form) with a specific center, you need to provide the center of the IC in the "Index" column. Indexes start at 0 for the first atom.

python main.py --mode test_with_IC --input your_input.csv > testwIC_your_input.out

## 4. Verbose Levels

Use the --verbose flag to control output detail:

--verbose 0: No details printed in the output (silent mode) --> Default

--verbose 1: Summary of predictions + Some cleaning details

--verbose 2: Detailed view of every deprotonation step

## 5. Python pipelines

If you prefer to call the model using Python lines, you can do so by using the predict() function directly:

from predict import predict

predicted_pkas, protonated_smiles = predict("your_dataset.csv", pH=7.4)

# 📖 Citation

If you use this code or model, please cite:

Genzling J, Luo Z, Weiser B, Moitessier N. Leveraging our Teacher’s Experience to Improve Machine Learning: Application to pKa Prediction. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-bpd53-v2 
This content is a preprint and has not been peer-reviewed.

# 🧠 Tips

Use Cheminfo SMILES viewer to visualize and debug SMILES (https://www.cheminfo.org/Chemistry/Cheminformatics/Smiles/index.html)

If protonation states are off, check atom indexing or consider using neutral forms.

You can retrain on your own dataset by modifying train_pKa_predictor.py.

# 🛠 Support

Feel free to reach out via email or GitHub issues if you need help using or adapting the model.



# MolGpKa
Fast and accurate prediction of the pKa values of small molecules is important in the drug discovery process since the ionization state of a drug has significant influence on its activity and ADME-Tox properties. MolGpKa is a tool for pKa prediction using graph-convolutional neural network model. The model works by learning pKa related chemical patterns automatically and building reliable predictors with learned features.

## Addendum installation
See the Docker_README.md file

## Requirements

* Python 3.6
* Pytorch >=1.4
* Pytorch-geometric (https://github.com/rusty1s/pytorch_geometric)
* RDKit (http://www.rdkit.org/docs/Install.html)
* py3Dmol 
* sklearn 0.21.3
* numpy 1.18.1
* pandas 0.25.3
* pickle

## Usage

### Using trained model for pKa prediction
`example.ipynb` is an example notebook for using MolGpKa, model weights file are located in `models`.

### Training model for convolutional-graph neural networks

1. `prepare_dataset_graph.py`--First, you should prepare the molecular file `mols.sdf` from ChEMBL database like the example. Then you will get two files `train.pickle, valid.pickle` in `datasets/` when you run the script  for data preparation.

2. `train_graph.py`--The purpose of this code is to train the graph-convolutional neural network model for pka prediction, the parameter file of MolGpKa will save in `models/`. You need to train the model for acidic ioniable center and basic ioniable center separately with corresponding data.

### Training model for AP-DNN

```
src/baseline/prepare_dataset_ap.py
src/baseline/train_ap.py
```
These scripts are designed to construct AP-DNN model which contain data preparation and model training.


## Benchmark set for pka substitution effects

In order to test the substitution effects extensively, we created a benchmark set by performing matched molecular pair analysis on experimental pKa data sets collected by Baltruschat et al. The benchmark set contains 4322 data points.
