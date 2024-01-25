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
import argparse
from tqdm import tqdm
import wandb
import pickle
import glob
import pandas as pd


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def seed_everything(seed = 1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(predict_data, gene_dim, drug_dim, model_file, hidden_folder, batch_size, result_file, cell_features, drug_features, perturb_table, test_result_df, cv, CUDA_ID):
    
    createFolder(hidden_folder)
    createFolder(result_file)

    feature_dim = gene_dim + drug_dim
    
    model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID) 
    predict_feature, predict_label = predict_data
    predict_label_gpu = predict_label.cuda(CUDA_ID)

    model.cuda(CUDA_ID)
    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)
    
    #Test
    test_predict = torch.zeros(0,0).cuda(CUDA_ID)
    tot_labels = torch.zeros(0,0).cuda(CUDA_ID)
    
    term_hidden_map = {}	

    batch_num = 0
    loss = nn.MSELoss()

    tot_test_loss = 0

    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as tstepoch:
            for i, (inputdata, labels) in enumerate(tstepoch):
                tstepoch.set_description(f"Test")
                # Convert torch tensor to Variable
                features = build_input_vector(inputdata, cell_features, drug_features)
        
                cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)
                cuda_labels = Variable(labels.cuda(CUDA_ID), requires_grad=False)
        
                # make prediction for test data
                aux_out_map, term_hidden_map = model(cuda_features, perturb_table)
        
                if test_predict.size()[0] == 0:
                    test_predict = aux_out_map['final'].data
                    tot_labels = cuda_labels
                else:
                    test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
                    tot_labels = torch.cat([tot_labels, cuda_labels], dim=0)
        
                for term, hidden_map in term_hidden_map.items():
                    hidden_file = hidden_folder+'/'+term+'_hidden.pickle'           
                    with open(hidden_file, 'wb') as f:
                        pickle.dump(hidden_map.data.cpu().numpy(), f, pickle.HIGHEST_PROTOCOL)
                
                test_loss = 0
                for name, output in aux_out_map.items():
                    if name == 'final':
                        test_loss = loss(output, cuda_labels)
                tot_test_loss += test_loss  
                labels_gpu = tot_labels
    
                test_loss_out = tot_test_loss/(i+1)
                test_pearson = pearson_corr(test_predict, labels_gpu)
                test_spearman = spearman_corr(test_predict, labels_gpu)
                test_rmse = rmse(test_predict, labels_gpu)
                test_ci = ci(test_predict, labels_gpu)
        
                print(f"batch_num: {batch_num}, test_loss: {test_loss_out:.6f}, test rmse: {test_rmse:.6f}, test pearson corr: {test_pearson:.6f}, test spearman corr: {test_spearman:.6f}, test ci: {test_ci:.6f}")
        
                log_dict = {'test/loss': tot_test_loss/(i+1), 'test/rmse': test_rmse, 'test/pearson': test_pearson, 'test/spearman': test_spearman, 'test/ci': test_ci, 'step': batch_num}
                wandb.log(log_dict)
    
                tstepoch.set_postfix(loss=test_loss_out.item())        
                batch_num += 1

    final_test_loss = tot_test_loss/len(test_loader)
    test_rmse = rmse(test_predict, tot_labels)
    test_pearson_corr = pearson_corr(test_predict, tot_labels)
    test_spearman_corr = spearman_corr(test_predict, tot_labels)
    test_ci = ci(test_predict, tot_labels)
    
    print(f"test_loss: {final_test_loss:.6f}, test rmse: {test_rmse:.6f}, test pearson corr: {test_pearson_corr:.6f}, test spearman corr: {test_spearman_corr:.6f}, test ci: {test_ci:.6f}")
    
    wandb.summary.update({
        'test_loss': final_test_loss,
        'test_rmse': test_rmse,
        'test_pearson': test_pearson_corr,
        'test_spearman': test_spearman_corr,
        'test_ci': test_ci
    })

    test_result_df['CV'].append(cv)
    test_result_df['loss'].append((final_test_loss).cpu().detach().numpy())
    test_result_df['rmse'].append(test_rmse)
    test_result_df['pearson'].append(test_pearson_corr.cpu().detach().numpy())
    test_result_df['spearman'].append(test_spearman_corr)
    test_result_df['ci'].append(test_ci)

    np.savetxt(result_file+'/drugcell.predict', test_predict.cpu().numpy(),'%.4e')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dcell')
    parser.add_argument('-data_type', help='type of training data', type=str, default='IC50')
    parser.add_argument('-cv', help='cross validation number', type=int, default=1)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=2)
    parser.add_argument('-load', help='Model file', type=str, default='MODEL/model_200')
    parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
    parser.add_argument('-result', help='Result file name', type=str, default='Result/')
    parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
    parser.add_argument('-netgp_top', type=int, default=100)
    parser.add_argument('-runname', help='run name on wandb', type=str)
    
    opt = parser.parse_args()
    torch.set_printoptions(precision=5)
    
    wandb.init(
        project='drug_response',
        name = opt.runname
              )
    wandb.config.update(opt)
    
    seed_everything()
    
    data_dir = "/data/project/yeojin/drug_response_2023/MyModel/data/mydata" # should be changed to where the data have been located
    
    gene2id = os.path.join(data_dir, 'gene2ind_final.txt')
    drug2id = os.path.join(data_dir, 'drug2ind_final.txt')
    cell2id = os.path.join(data_dir, 'cell2ind_final.txt')
    
    genotype = os.path.join(data_dir, 'cell2exp_final.txt')
    fingerprint = os.path.join(data_dir, 'drug2fingerprint_2048.txt')
    
    
    test_result_df = {}
    keys = ['CV', 'loss', 'rmse', 'pearson', 'spearman', 'ci']
    
    for key in keys:
        test_result_df[key] = []
    
    
    print(f"# ======= CV: {opt.cv} ======= #")
    test_dir = os.path.join(data_dir, f'model_input/{opt.data_type}_cell_split/test_{opt.data_type}_cell_cv{opt.cv}.txt')
    
    CUDA_ID = opt.cuda
    
    # load input data
    predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(test_dir, cell2id, drug2id)
    gene2id_mapping = load_mapping(gene2id)
    
    def top_n_to_1(row, n= opt.netgp_top):
        sorted_row = np.argsort(row)
        row[sorted_row[:-n]] = 0
        row[sorted_row[-n:]] = 1
        return row
    
    netGP = pd.read_csv('data/mydata/netGP_profile.out', sep='\t').drop(columns=['drug_name']) # should be changed to where the data have been located
    netGP_col = netGP.columns[1:]
    netGP[netGP_col] = netGP[netGP_col].apply(top_n_to_1, axis=1)
    
    perturb_table = torch.tensor(netGP.drop(columns=['drug_idx']).values)
    
    
    # load cell/drug features
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(fingerprint, delimiter=',')
    
    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    num_genes = len(gene2id_mapping)
    drug_dim = len(drug_features[0,:])
    
    
    print("Total number of genes = %d" % num_genes)
    
    createFolder(opt.result)
    
    result_dir = os.path.join(opt.result, f'cv_{opt.cv}')
    hidden_dir = os.path.join(opt.hidden, f'cv_{opt.cv}')
    
    load_dir = glob.glob(f'/data/project/yeojin/drug_response_2023/MyModel/{opt.load}/cv_{opt.cv}/model_best_*.pt')
    # load_dir = f'/data/project/yeojin/drug_response_2023/MyModel/{opt.load}/cv_{opt.cv}/model_50.pt'
    
    predict(predict_data, num_genes, drug_dim, load_dir[0], hidden_dir, opt.batchsize, result_dir, cell_features, drug_features, perturb_table, test_result_df, opt.cv, CUDA_ID)	
    
    # for key, value_list in test_result_df.items():
    #     if key == 'CV':
    #         value_list.append('mean')
    #     else:
    #         mean_value = sum(value_list) / len(value_list)
    #         value_list.append(mean_value)
    
    test_result_df = pd.DataFrame(test_result_df)
    filename = os.path.join(result_dir, f'total.result')
    test_result_df.to_csv(filename, sep='\t', header=True, index=False)






