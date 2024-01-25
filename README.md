# DrugPT-Net
![](https://github.com/syjssj95/DrugPT-Net/blob/main/image/model_architecture.png)
DrugPT-Net is a Drug PerTurbation guided visible neural Network for drug response prediction at transcriptomic level, which incorporates the gene ontology-based VNN with gene perturbation effects at transcriptomic level. Specifically, we use continuous gene expression values as embedding vectors via the Radial Basis Function (RBF) kernel. To incorporate biological mechanisms perturbed by drugs into the VNN, we (1) integrate information of genes associated with drug-affected biological pathways simulated on the PPI network into gene expression embeddings, and (2) employ a drug-aware gating function, which selectively retains or forgets information downstream over the ontology hierarchy based on its importance to the corresponding molecular embedding of drug.

# Environment Setup
The following has been used for running DrugPT-Net model:
```
Cuda Version 12.1
Miniconda 23.7.4
Python >=3.8
pytorch 2.1.0
pytorch-cuda 12.1
scipy 1.11.1
pandas 2.0.3
wandb 0.15.10
networkx 2.7.1
scikit-learn 1.2.2
lifelines 0.27.8
numpy 1.25.2
```
The environment used for training our pre-trained model is provided in _DrugPT-Net.yaml_ file.
If you are creating a virtual environment through the provided file, run the following command line:
```
conda env create -f DrugPT-Net.yaml
```

# Data
1. Input file
   *__Drug response data__*: data/model_input/IC50_revised/[train/valid/test]_IC50_[random/cell]_cv[1-10].txt
   *__Gene expression data__*: data/cell2exp_[final/standardscale].txt
   *__Cell line index file__*: data/cell2ind_final.txt
   *__Gene index file__*: data/gene2ind_final.txt
   *__Drug index file__*: data/gene2ind_final.txt
   *__Ontology graph file__*: data/ontology_final.txt
   *__Drug fingerprint embedded file__*: data/drug2fingerprint_2048.txt
     
2. Drug-target interaction information file
   *__Perturbation score output file__*: data/NetGP_profile.out
   *__Drug smiles data__*: data/drug_smiles.tsv
   *__Drug target information file__*: data/drug_target_info.tsv

For changing the drug data, you may visit https://github.com/minwoopak/NetGP to calculate the new perturbation score result. The obtained result file may substitute the _data/NeGP_profile.out_ file. 

A sample ontology file _ontology_samply.txt_ can be used to substitute _data/ontology_final.txt_ for a quick view of model structure for better understanding of model running procedure.

# Running Procedure
## 1. Training Model
For training DrugPT-Net, run the following code:
```
python -u code/train.py \
-data_type IC50 \
-cv 1 \
-epoch 100 \
-lr 0.01 \
-batchsize 512 \
-cuda 0 \
-seed 1234 \
-genotype_hiddens 6 \
-drug_hiddens '128,64,50,6' \
-final_hiddens 6 \
-gamma 0.1 \
-gene_embed_dim 16 \
-netgp_top 200 \
-test_batchsize 512 \
-runname <directory name for the trained model>
```
1. _-genotype_hiddens_: the number of neurons in each term in genotype parts

2. _-drug_hiddens_: the number of neurons in each layer for embedding drug data

3. _-final_hiddens_: the number of neurons in the top layer

4. _-gene_embed_dim_: embedded dimension for each gene

5. _-netgp_top_: the number of top-scores that would be counted for perturbation table creation

You may perform training as well as testing by running this code. If you wish to perform training only, you may modify _code/train.py_ file to turn off the testing procedure.

You may turn on wandb for logging your results. 

## 2. Testing Model Prediction
If you wish to test the trained model, run the following code:
```
python -u code/predict.py \
-cv 5 \
-batchsize 512 \
-load <file name for the trained model that would be loaded for testing> \
-hidden <directory name for the output of hidden results> \
-result <directory name for the final prediction results> \
-cuda 0 \
-netgp_top 200 \
-runname  <directory name for running the prediction>
```
1. _-load_: file name for the trained model that would be loaded for testing

2. _-hidden_: directory in which each GO term values for prediction is saved

3. _-result_: directory in which final results are saved

Because training of this model takes considerable amount of time, you may find a pretrained model in  _pretrained_model/pre-trained_model_cv10.tar.gz_ file. You may unzip this file to use for testing. The final results for prediction can be found in the folder specified by _-result_.













