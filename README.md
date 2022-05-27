# Translating from Proteins to Ribonucleic Acids for Ligand-Binding Site Detection


Goals of the project:\
(i) Prepare RNA-ligand dataset that contains ligands with drug-like properties and is suitable for the machine-learning-based detection of binding sites\
(ii) Introduce methodology that considers 3D structural features of binding site for dataset splitting and thereby minimizes test-set leakage\
(iii) Detect ligand-binding sites in RNA structures using 3D-CNNs and investigate the influence of protein-based pre-training on the binding site detection performance\
\
The project is described in the following paper:\
L. Moeller, L. Guerci, C. Isert, K. Atz, G. Schneider, ***Mol. Inf.*** **2022**\
Please cite this paper when using the RNA dataset, models or methods introduced in this work.

---

## Setup for local usage
Note that the all functions were tested with python 3.8.3 and a Linux operating system.
### 1. Download required software
```
git clone https://github.com/lkmoeller/rna_binding.git
cd rna_binding
```
### 2. Setup conda environment
```
conda env create -f environment.yml
```
In order to perform RNA dataset splitting, the tool QML need to be installed manually (https://www.qmlcode.org/installation.html).
### 3. Adjust python path variable
Add path to `rna_binding` directory to python path variable by adding the following line to `~/.bashrc` (Linux, replace directory by the path that contains the cloned directories)
```
export PYTHONPATH="/path/to/rna_binding:$PYTHONPATH"
```

---

## RNA dataset preparation
### Download of RNA dataset
(i) Download PDB batch download script  (https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script) and ensure that script has correct execution permissions.\
(ii) Download PDB files of interest by using the PDB batch download script. Files included in the RNA dataset generated in this work are listed in `rna_binding/rna_binding/data/mol/rna/pdb/pdb_ids.txt`.\
(iii) Download NDB files of interest (http://ndbserver.rutgers.edu/). Files included in the RNA dataset generated in this work are listed in `rna_binding/rna_binding/data/mol/rna/pdb/ndb_ids.txt` for manual download.\
(iv) Move all PDB and NDB files to the directory `rna_binding/rna_binding/data/mol/rna/pdb`.


### RNA data filtering
(i) Download file "Components-smiles-stereo-oe.smi" (http://ligand-expo.rcsb.org/ld-download.html) from pdb and store in directory `rna_binding/rna_binding/data/dataset_preparation`.\
(ii) Prepare RNA dataset:
```
python data_preparation/rna_data_prep.py
```
Filtering not required if RNA dataset generated in this work used.

---

## RNA dataset splitting
(i) Extract exemplary binding sites for analysis:
```
python data_splitting/binding_site_extraction.py
```
(ii) Cluster binding sites (only file that required QML to be installed):
```
python data_splitting/clustering.py -filter 1 -optimize 1
```
(iii) Split dataset:
```
python data_splitting/rna_split.py
```
Note that `fold_dict` needs to be adjusted according to the clustering results before running the dataset splitting script. Dataset splitting and adjustment of `fold_dict` not required if RNA dataset generated in this work used (`rna_binding/rna_binding/data/crossval/rna/all`).

---

## Example usecase: detection of druggable RNA binding sites with 3D-CNNs
### Data preparation, splitting, and descriptor generation
(i) Download scPDB database for pre-training with protein data (http://bioinfo-pharma.u-strasbg.fr/scPDB/) to the directory `rna_binding/rna_binding/data/mol/prot/scpdb`. Filter and split protein dataset:
```
python data_preparation/prot_data_prep.py
python data_splitting/prot_split.py
```
(ii) Generate grid-descriptors:
```
python descriptor_generation/descriptor_calculation.py -mode 0 -path path/to/rna/file/ids
python descriptor_generation/descriptor_calculation.py -mode 1 -path path/to/prot/file/ids
```
For multisubmission, use the script `descriptor_generation/multisub_descriptor.py`

### Model training
(i) Protein-based pre-training:
```
python training/pretrain.py -nn RNet1 -bs 4 -nw 4 -save_int 10 -lr 1e-5 -opt_mode all
```
(ii) Training solely on RNA data. Execute for all folds or use multisubmission (`training/multisub_finetune.py`).
```
python training/finetune.py -nn RNet1 -bs 4 -nw 4 -start_epochs 0 -end_epochs 4001 -save_int 10 -lr 1e-5 -opt_mode all -fold 1
```
Models solely trained on RNA data can be found in the directory `rna_binding/rna_binding/data/models/RNet1/rna_only/l5_bs4_all`.\
(iii) Fine-tuning using RNA data:
```
python training/finetune.py -nn RNet1 -bs 4 -nw 4 -start_epochs 0 -end_epochs 4001 -save_int 10 -lr 1e-5 -opt_mode all -fold 1 -model_path data/models/RNet1/pretraining/l5_bs4_all/model_best.pkl
```
Models pre-trained on protein and fine-tuned on RNA data can be found in the directory `rna_binding/rna_binding/data/models/RNet1/finetuning/l5_bs4_all`.

### Model performance evaluation
(i) Generate predictions and evaluate model performance. Execute for all conditions (rna_only, finetuning) and folds or use multisubmission (`prediction/multisub_predict.py`).
```
python prediction/predict.py -add_name _fold1 -model_path data/models/RNet1/finetuning/l5_bs4_all/model_fold1_best.pkl -data_path data/crossval/rna/all/test_fold1.csv
```
(ii) To generate visualizations (cube-files) of predicted and ground truth binding sites, run prediction script using `-save_vis 1`.
