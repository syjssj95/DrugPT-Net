import sys
import os
import random
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugpt import *
from predict import *
import argparse
import numpy as np
import time
from tqdm import tqdm
import wandb
import pickle
from torch import tensor
import pandas as pd
from pytorchtools import EarlyStopping
from torch.optim import lr_scheduler


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)



def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

        term_mask_map[term] = mask_gpu

    return term_mask_map


def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features, gamma, gene_embed_dim, perturb_table, node_num):

    epoch_start_time = time.time()
    best_model_epoch = 0
    min_loss = 100000
    patience = 20
    
    createFolder(model_save_folder)
    
    # drugpt neural network
    model = drugpt(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, gamma, gene_embed_dim)
    

    train_feature, train_label, test_feature, test_label = train_data

    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

    model.cuda(CUDA_ID)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=True)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

    loss = nn.MSELoss()
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    early_stopping = EarlyStopping(patience = patience, verbose = True)

    for epoch in range(train_epochs):

        #Train
        model.train()
        train_predict = torch.zeros(0,0).cuda(CUDA_ID)
        tot_labels = torch.zeros(0,0).cuda(CUDA_ID)
        total_loss_train = 0
        total_loss_val = 0
        
        with tqdm(train_loader, unit='batch') as tepoch:
            for i, (inputdata, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1} train")

                # Convert torch tensor to Variable
                features = build_input_vector(inputdata, cell_features, drug_features)

                cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
                cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                aux_out_map, _ = model(cuda_features, perturb_table)

                if train_predict.size()[0] == 0:
                    train_predict = aux_out_map['final'].data
                    tot_labels = cuda_labels
                else:
                    train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
                    tot_labels = torch.cat([tot_labels, cuda_labels], dim=0)

                loss_train = 0
                for name, output in aux_out_map.items():
                    if name == 'final':
                        loss_train += loss(output, cuda_labels)
                    else:
                        loss_train += 0.2 * loss(output, cuda_labels)
                total_loss_train += loss_train

                if i%10 == 0:
                    log_dict = {'train/loss': total_loss_train/(i+1), 'train/rmse': rmse(train_predict, tot_labels), 'train/pearson': pearson_corr(train_predict, tot_labels), 'train/spearman': spearman_corr(train_predict, tot_labels), 'train/ci': ci(train_predict, tot_labels), 'epoch': epoch, 'step': i}
                    wandb.log(log_dict)

                loss_train.backward()

                for name, param in model.named_parameters():
                    if '_direct_gene_layer.weight' not in name:
                        continue
                    term_name = name.split('_')[0]
                    param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

                optimizer.step()               
                tepoch.set_postfix(loss=loss_train.item())

        train_pearson = pearson_corr(train_predict, tot_labels)
        train_spearman = spearman_corr(train_predict, tot_labels)
        train_loss = total_loss_train/len(train_loader)
        train_rmse = rmse(train_predict, tot_labels)
        train_ci = ci(train_predict, tot_labels)

        if epoch % 30 == 0:
            torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

        #Test: random variables in training mode become static
        model.eval()

        test_predict = torch.zeros(0,0).cuda(CUDA_ID)
        test_tot_labels = torch.zeros(0,0).cuda(CUDA_ID)

        with torch.no_grad():
            with tqdm(test_loader, unit='batch') as vepoch:
                for i, (inputdata, labels) in enumerate(vepoch):
                    vepoch.set_description(f"Epoch {epoch+1} val")
                    
                    # Convert torch tensor to Variable
                    features = build_input_vector(inputdata, cell_features, drug_features)
                    cuda_features = Variable(features.cuda(CUDA_ID))
                    cuda_labels = Variable(labels.cuda(CUDA_ID))
    
                    aux_out_map, _ = model(cuda_features, perturb_table)

                    if test_predict.size()[0] == 0:
                        test_predict = aux_out_map['final'].data
                        test_tot_labels = cuda_labels
                    else:
                        test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
                        test_tot_labels = torch.cat([test_tot_labels, cuda_labels], dim=0)
                    
                    loss_val = 0
                    for name, output in aux_out_map.items():
                        if name == 'final':
                            loss_val = loss(output, cuda_labels)
                    total_loss_val += loss_val
                       
                    if i%10 == 0:
    
                        log_dict = {'val/loss': total_loss_val/(i+1), 'val/rmse': rmse(test_predict, test_tot_labels), 'val/pearson': pearson_corr(test_predict, test_tot_labels), 'val/spearman': spearman_corr(test_predict, test_tot_labels), 'val/ci': ci(test_predict, test_tot_labels), 'epoch': epoch, 'step': i}
                        wandb.log(log_dict)

        test_pearson = pearson_corr(test_predict, test_tot_labels)
        test_spearman = spearman_corr(test_predict, test_tot_labels)
        test_loss = total_loss_val/len(test_loader)
        test_rmse = rmse(test_predict, test_tot_labels)
        test_ci = ci(test_predict, test_tot_labels)
        
        vepoch.set_postfix(accuracy=test_pearson)

        epoch_end_time = time.time()


        print("epoch\t%d\tcuda_id\t%d\telapsed_time\t%s\ntrain_pearson\t%.6f\ttrain_spearman\t%.6f\ttrain_rmse\t%.6f\ttrain_ci\t%.6f\ttrain_loss\t%.6f\ntest_pearson\t%.6f\ttest_spearman\t%.6f\ttest_rmse\t%.6f\ttest_ci\t%.6f\ttest_loss\t%.6f" % (epoch, CUDA_ID, epoch_end_time-epoch_start_time, train_pearson, train_spearman, train_rmse, train_ci, train_loss, test_pearson, test_spearman, test_rmse, test_ci, test_loss))
        epoch_start_time = epoch_end_time

        if min_loss > test_loss:
            min_loss = test_loss
            best_model_epoch = epoch
            best_model_save = model

        step_lr_scheduler.step()        
            
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print(f'Model early stopped with patience {patience} at epoch {epoch}')
            break
    
    
    torch.save(best_model_save, model_save_folder + '/model_best_' + str(best_model_epoch) + '.pt')
    torch.save(model, model_save_folder + '/model_final.pt')

    wandb.summary.update({
        'train_total_loss': train_loss,
        'train_pearson': train_pearson,
        'train_spearman': train_spearman,
        'train_rmse': train_rmse,
        'train_ci': train_ci,
        'test_loss': test_loss,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
        'test_rmse': test_rmse,
        'test_ci': test_ci,
        'best performed model (epoch)': best_model_epoch
    })

    print("Best performed model (epoch)\t%d" % best_model_epoch)


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-data_type', help='type of training data', type=str, default='IC50')
parser.add_argument('-cv', help='cross validation number', type=int, default=1)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-seed', type=int, default=1234)
parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=6)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,6')
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=6)
parser.add_argument('-gamma', help='Gamma value for gene expression vectorizing', type=str, default=0.1)
parser.add_argument('-gene_embed_dim', type=int, default=128)
parser.add_argument('-netgp_top', type=int, default=200)
parser.add_argument('-runname', help='run name on wandb', type=str)

# for testing
parser.add_argument('-test_batchsize', help='Batchsize', type=int, default=512)


opt = parser.parse_args()
torch.set_printoptions(precision=5)

runname = f'cv{opt.cv}_{opt.runname}'

wandb.init(
    project='drug_response',
    name = runname
      )
wandb.config.update(opt)

seed_everything()

data_dir = "/data/project/yeojin/drug_response_2023/MyModel/data/mydata" # should be changed to where the data has been located

gene2id = os.path.join(data_dir, 'gene2ind_final.txt')
drug2id = os.path.join(data_dir, 'drug2ind_final.txt')
cell2id = os.path.join(data_dir, 'cell2ind_final.txt')

genotype = os.path.join(data_dir, 'cell2exp_final.txt')
fingerprint = os.path.join(data_dir, 'drug2fingerprint_2048.txt')

onto = os.path.join(data_dir, 'drugcell_ont_final.txt')


print(f"# ======= CV: {opt.cv} Train ======= #")
train_dir = os.path.join(data_dir, f'model_input/{opt.data_type}_cell_split/train_{opt.data_type}_cell_cv{opt.cv}.txt')
val_dir = os.path.join(data_dir, f'model_input/{opt.data_type}_cell_split/valid_{opt.data_type}_cell_cv{opt.cv}.txt')

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(train_dir, val_dir, cell2id, drug2id)
gene2id_mapping = load_mapping(gene2id)

# load cell/drug features
cell_features = np.genfromtxt(genotype, delimiter=',')
drug_features = np.genfromtxt(fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

gamma = float(opt.gamma)
gene_embed_dim = opt.gene_embed_dim

def top_n_to_1(row, n= opt.netgp_top):
    sorted_row = np.argsort(row)
    row[sorted_row[:-n]] = 0
    row[sorted_row[-n:]] = 1
    return row

# create perturbation table from NetGP result 
netGP = pd.read_csv('data/mydata/netGP_profile.out', sep='\t') # should be changed to where the data have been located
netGP_col = netGP.columns[2:]
netGP[netGP_col] = netGP[netGP_col].apply(top_n_to_1, axis=1)

drug_target = pd.read_csv('data/mydata/drug_target_info.tsv', sep='\t') # should be changed to where the data have been located
for i, row in drug_target.iterrows():
    if row.gene_name in netGP_col:
        netGP.loc[netGP['drug_name'] == row.drug_name, row.gene_name] = 2

perturb_table = torch.tensor(netGP.drop(columns=['drug_idx', 'drug_name']).values)


# load ontology
dG, root, term_size_map, term_direct_gene_map, node_num = load_ontology(onto, gene2id_mapping)

num_hiddens_genotype = opt.genotype_hiddens
num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))
num_hiddens_final = opt.final_hiddens

CUDA_ID = opt.cuda

createFolder(f'Model/train_cvdata_{opt.runname}')

model_dir = f'Model/train_cvdata_{opt.runname}/cv_{opt.cv}'

train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, model_dir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features, gamma, gene_embed_dim, perturb_table, node_num)

###########################################
# DrugPT testing (prediction)

test_result_df = {}
keys = ['CV', 'loss', 'rmse', 'pearson', 'spearman', 'ci']

for key in keys:
    test_result_df[key] = []

print(f"# ======= CV: {opt.cv} Test ======= #")
test_dir = os.path.join(data_dir, f'model_input/{opt.data_type}_cell_split/test_{opt.data_type}_cell_cv{opt.cv}.txt')

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(test_dir, cell2id, drug2id)

createFolder(f'results/Hidden_cvdata_{opt.runname}')
createFolder(f'results/Result_cvdata_{opt.runname}')

result_dir = f'results/Result_cvdata_{opt.runname}/cv_{opt.cv}'
hidden_dir = f'results/Hidden_cvdata_{opt.runname}/cv_{opt.cv}'

load_dir = glob.glob(f'/data/project/yeojin/drug_response_2023/MyModel/{model_dir}/model_best_*.pt')

predict(predict_data, num_genes, drug_dim, load_dir[0], hidden_dir, opt.test_batchsize, result_dir, cell_features, drug_features, perturb_table, test_result_df, opt.cv, CUDA_ID)

test_result_df = pd.DataFrame(test_result_df)
filename = os.path.join(result_dir, f'total.result')
test_result_df.to_csv(filename, sep='\t', header=True, index=False)


