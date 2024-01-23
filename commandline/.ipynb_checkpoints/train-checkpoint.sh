#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
traindatafile=$inputdir"drugcell_train.txt"
valdatafile=$inputdir"drugcell_val.txt"
ontfile=$inputdir"drugcell_ont.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

cudaid=0

modeldir=Model_sample
mkdir $modeldir

source activate pytorch3drugcell

python -u ../code/train_drugcell.py -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -epoch 100 -batchsize 5000 > train_sample.log



python -u code/train_drugcell.py -onto data/drugcell_ont.txt -train data/drugcell_train.txt -test data/drugcell_val.txt -epoch 100 -lr 0.001 -batchsize 5000 -model Model -cuda 0 -gene2id data/gene2ind.txt -drug2id data/drug2ind.txt -cell2id data/cell2ind.txt -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -genotype data/cell2mutation.txt -fingerprint data/drug2fingerprint.txt -runname $runname_on_wandb > train_sample.log
