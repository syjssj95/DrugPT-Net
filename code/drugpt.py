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

'''
The code for the creation of ontology graph network has been largely
inspired by DrugCell code by Kuenzi et.
'''


class drugpt(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, gamma, gene_embed_dim):

        super(drugpt, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map   

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)
        
        self.gene_dim = ngene			   
        self.drug_dim = ndrug

        # gamma value for rbf kernel
        self.gamma = gamma

        # embedded dimension for each gene: default 16
        self.gene_embed_dim = gene_embed_dim

        # vectorize each gene expression scalar values
        self.gene_exp_vector_embed()

        # embed lookup table of the corresponding drug
        self.lookup_table_embed()

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs	
        self.construct_NN_drug()
        
        final_input_size = self.gene_embed_dim + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final,1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    
    def rbf_kernel(self, centers, gamma):
        vect = torch.zeros(centers.shape[0], centers.shape[1], centers.shape[1])
        for i in range(centers.shape[0]):
            gene = centers[i].view(1, -1)
            gene_transf = torch.reshape(gene, (-1, 1))
            vect[i] = torch.exp(-gamma * torch.square(gene_transf - gene)).float()
        return vect
    

    # vectorize gene expression scalar values
    def gene_exp_vector_embed(self):
        self.add_module('RBF_embedding_layer', nn.Linear(self.gene_dim, self.gene_embed_dim))
    

    # construct lookup table of the corresponding drug
    def create_lookup_table(self, table, idx_lst):
        lookup_table = torch.zeros(len(idx_lst), table.shape[1])
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
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                
                self.add_module(term+'_gate_layer', nn.Linear(self.gene_embed_dim+self.num_hiddens_drug[-1], 1))
                
                self.add_module(term+'_aux_linear_layer1', nn.Linear(self.gene_embed_dim,1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(1,1))

            dG.remove_nodes_from(leaves)


    def forward(self, x, table):
        gene_input = x.narrow(1, 0, self.gene_dim)
        
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)
        drug_idx = x.narrow(1, self.gene_dim+self.drug_dim, 1)

        # lookup table creation and vectorization
        lookup_table = self.create_lookup_table(table, drug_idx).cuda(x.device)
        lookup_table_vect = self._modules['lookup_table_embedding_layer'](lookup_table)
    
        # gene expression scalar value vector embedding
        gene_input_rbf = self.rbf_kernel(gene_input, self.gamma)  
        gene_input_vect = self._modules['RBF_embedding_layer'](gene_input_rbf.cuda(x.device))

        # addition of two values to finally create final input
        gene_input_final = torch.add(gene_input_vect, lookup_table_vect)

        gene_input_final_permute = gene_input_final.permute(0, 2, 1)

        # define forward function for drug dcell #################################################
        drug_out = drug_input

        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)]( torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))	

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input_final_permute)
        
        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child].permute(0, 2, 1))

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])
                        
                child_input = torch.cat(child_input_list,2)
                term_NN_out = self._modules[term+'_linear_layer'](child_input).permute(0, 2, 1)

                Tanh_out = torch.tanh(term_NN_out)
                batchnorm_out = self._modules[term+'_batchnorm_layer'](Tanh_out)

                # applying drug-aware gating function to each GO term
                gate_out = torch.sigmoid(self._modules[term+'_gate_layer'](torch.cat((batchnorm_out.mean(dim=1), drug_out), 1)))
                term_NN_out_map[term] = gate_out.unsqueeze(1)*batchnorm_out
                
                mean_out = term_NN_out_map[term].mean(dim=1)
                aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](mean_out))
                
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)	

        # connect two neural networks at the top #################################################
            
        root_out = term_NN_out_map[self.root].mean(dim=1)
        
        final_input = torch.cat((root_out, drug_out), 1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map
