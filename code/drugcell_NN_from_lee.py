import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
import pickle


class drugcell_nn_from_lee(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, gamma, gene_embed_dim, CUDA_ID):

        super(drugcell_nn_from_lee, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map   

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)		   

        # ngenes, gene_dim are the number of all genes	
        self.gene_dim = ngene			   
        self.drug_dim = ndrug
        
        self.gamma = gamma
        self.gene_embed_dim = gene_embed_dim
        self.CUDA_ID = CUDA_ID
        
        self.gene_exp_vector_embed()
        self.lookup_table_embed()

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs	
        self.construct_NN_drug()

        # add modules for final layer
        final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final,1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    
    def rbf_kernel(self, centers, gamma, CUDA_ID):
        vect = torch.zeros(centers.shape[0], centers.shape[1], centers.shape[1]).cuda(CUDA_ID)
    #     print(vect.shape)
        for i in range(centers.shape[0]):
            gene = centers[i].view(1, -1)
    #         print(gene.shape)
            gene_transf = torch.reshape(gene, (-1, 1))
    #         print(gene_transf.shape)
            vect[i] = torch.exp(-gamma * torch.square(gene_transf - gene)).float()
        return vect
    
    
    def gene_exp_vector_embed(self):
        self.add_module('RBF_embedding_layer', nn.Linear(self.gene_dim, self.gene_embed_dim))
    
    
    def create_lookup_table(self, table, idx_lst, CUDA_ID):
        lookup_table = torch.zeros(len(idx_lst), table.shape[1]).cuda(CUDA_ID)
        idx_lst = list(map(int, idx_lst))
        
        for i in range(len(idx_lst)):
            lookup_table[i] = table[idx_lst[i]]
        return lookup_table.to(torch.int)
            
    
    def lookup_table_embed(self):
        self.add_module('lookup_table_embedding_layer', nn.Embedding(self.gene_dim, self.gene_embed_dim))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            # print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output


    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes 		
            self.add_module(term+'_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))


    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        for i in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(i+1), nn.Linear(input_size, self.num_hiddens_drug[i]))
            self.add_module('drug_batchnorm_layer_' + str(i+1), nn.BatchNorm1d(self.num_hiddens_drug[i]))
            self.add_module('drug_aux_linear_layer1_' + str(i+1), nn.Linear(self.num_hiddens_drug[i],1))
            self.add_module('drug_aux_linear_layer2_' + str(i+1), nn.Linear(1,1))

            input_size = self.num_hiddens_drug[i]


    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []   # term_layer_list stores the built neural network 
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term	
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            #leaves = [n for n,d in dG.out_degree().items() if d==0]
            #leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                
                self.add_module(term+'_term_context_layer', nn.Linear(term_hidden, term_hidden))
                self.add_module(term+'_batchnorm_layer2', nn.BatchNorm1d(term_hidden))
                
                self.add_module(term+'_dimension_layer', nn.Linear(self.gene_embed_dim, term_hidden))
                self.add_module(term+'_batchnorm_layer3', nn.BatchNorm1d(term_hidden))
                
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden,1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(1,1))

            dG.remove_nodes_from(leaves)


    # definition of forward function
    def forward(self, x, table):
        gene_input = x.narrow(1, 0, self.gene_dim)
#         print('gene_input shape:', gene_input.shape)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)
        drug_idx = x.narrow(1, self.gene_dim+self.drug_dim, 1)
#         print('drug_idx shape:', drug_idx.shape)
        
        # gene expression scalar value vector embedding
        gene_input_rbf = self.rbf_kernel(gene_input, self.gamma, self.CUDA_ID)
#         print(gene_input_rbf)
#         print('through rbf kernel:', gene_input_rbf.shape)

        gene_input_vect = self._modules['RBF_embedding_layer'](gene_input_rbf)
#         print('finally vectorized:', gene_input_vect.shape)
        
        # lookup table creation
        # table: lookup table을 만들기 위한 matrix
        lookup_table = self.create_lookup_table(table, drug_idx, self.CUDA_ID)
#         print('lookup_table:', lookup_table.shape)
        
        lookup_table_vect = self._modules['lookup_table_embedding_layer'](lookup_table)
#         print('lookup_table embeeded:', lookup_table_vect.shape)
        
        # addition of two values to finally create final input
        # name: gene_input_final
        gene_input_final = torch.add(gene_input_vect, lookup_table_vect)
        
#         print('final input shape:', gene_input_final.shape)
        
        gene_input_final_permute = gene_input_final.permute(0, 2, 1)
        

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input_final_permute)
#             print(f'{term}_direct_gene_layer: {term_gene_out_map[term].shape}')
        
        term_NN_out_map = {}
        aux_out_map = {}
        term_NN_out_map_3d = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map_3d[child].permute(0, 2, 1))

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])
                
                
#                 print(child_input_list)
                
#                 if i != 0:
#                     with open(f'child_input_list.pickle', 'wb') as f:
#                         pickle.dump(child_input_list, f, pickle.HIGHEST_PROTOCOL)
#                         print(sfsd)
                        
                child_input = torch.cat(child_input_list,2)

                term_NN_out = self._modules[term+'_linear_layer'](child_input).permute(0, 2, 1)
#                 print(f'{term}_linear_layer: {term_NN_out.shape}')
                term_NN_batchnorm = self._modules[term+'_batchnorm_layer1'](term_NN_out)

                term_context_out = self._modules[term+'_term_context_layer'](term_NN_batchnorm.permute(0, 2, 1)).permute(0, 2, 1)
                term_NN_out_map_3d[term] = self._modules[term+'_batchnorm_layer2'](term_context_out)
    
                mean_out = term_NN_out_map_3d[term].mean(axis=1)
        
                term_dimension_out = self._modules[term+'_dimension_layer'](mean_out)

                Tanh_out = torch.tanh(term_dimension_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer3'](Tanh_out)
#                 print(f'{term}_batchnorm_layer: {term_NN_out_map[term].shape}')
                
                aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
        
#                 print(f'{term}_aux_layer_1_out: {aux_layer1_out.shape}')
                
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)
#                 print(f'{term}_aux_linear_layer2: {aux_out_map[term].shape}')

        # define forward function for drug dcell #################################################
        drug_out = drug_input

        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)]( torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_'+str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules['drug_aux_linear_layer1_'+str(i)](drug_out))
            aux_out_map['drug_'+str(i)] = self._modules['drug_aux_linear_layer2_'+str(i)](aux_layer1_out) 		

        # connect two neural networks at the top #################################################
        final_input = torch.cat((term_NN_out_map[self.root], drug_out), 1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map
