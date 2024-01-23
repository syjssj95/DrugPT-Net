#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
testdatafile=$inputdir"drugcell_test.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

modelfile="../pretrained_model/drugcell_v1.pt"

resultdir="Result_sample"
hiddendir="Hidden_sample"

cudaid=$1

if [$cudaid = ""]; then
	cudaid=0
fi

mkdir $resultdir
mkdir $hiddendir

source activate pytorch3drugcell


python -u code/predict_drugcell.py -gene2id data/gene2ind.txt -cell2id data/cell2ind.txt -drug2id data/drug2ind.txt -genotype data/cell2mutation.txt -fingerprint data/drug2fingerprint.txt -hidden results/Hidden_baseline_orgdata_bsz_4 -result results/Result_baseline_orgdata_bsz_4 -predict data/drugcell_test.txt -load pretrained_model/drugcell_v1.pt -cuda 0


python -u code/predict_drugcell.py -predict data/drugcell_test.txt -batchsize 4 --gene2id data/gene2ind.txt -cell2id data/cell2ind.txt -drug2id data/drug2ind.txt -load pretrained_model/drugcell_v1.pt -hidden results/Hidden_baseline_orgdata_bsz_4 -result results/Result_baseline_orgdata_bsz_4 -cuda 0 -genotype data/cell2mutation.txt -fingerprint data/drug2fingerprint.txt -project baseline_orgdata_drug_fingerprint_bsz_4 > test_baseline_orgdata_drug_fingerprint_bsz_4.log