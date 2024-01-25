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




# Running Procedure

