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

from model_helper import *


class transformer(nn.Sequential):
    def __init__(self, transformer_emb_size_drug, transf_dropout_rate, transf_n_layer, transf_intermediate_size, transf_n_att_heads, transf_att_prob_dropout, transf_hidden_dropout):
        super(transformer, self).__init__()
        input_dim_drug = 2584
        self.transformer_emb_size_drug = transformer_emb_size_drug
        self.transformer_dropout_rate = transf_dropout_rate
        self.transformer_n_layer_drug = transf_n_layer
        self.transformer_intermediate_size_drug = transf_intermediate_size
        self.transformer_num_attention_heads_drug = transf_n_att_heads
        self.transformer_attention_probs_dropout = transf_att_prob_dropout
        self.transformer_hidden_dropout_rate = transf_hidden_dropout

        self.emb = Embeddings(input_dim_drug,
                         self.transformer_emb_size_drug,
                         61,
                         self.transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(self.transformer_n_layer_drug,
                                         self.transformer_emb_size_drug,
                                         self.transformer_intermediate_size_drug,
                                         self.transformer_num_attention_heads_drug,
                                         self.transformer_attention_probs_dropout,
                                         self.transformer_hidden_dropout_rate)
    def forward(self, v):
        e = v[0].long().to(v[0].device)
        e_mask = v[1].long().to(v[1].device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        # print(encoded_layers.shape)
#         print('from transformer:', encoded_layers[:,0].shape)
        return encoded_layers[:, 0]



class drugcell_nn_tta(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, num_hiddens_genotype, drug_embed_size, transf_dropout_rate, transf_n_layer, transf_intermediate_size, transf_n_att_heads, transf_att_prob_dropout, transf_hidden_dropout, num_hiddens_final, gamma, gene_embed_dim, topN_else_mask):

        super(drugcell_nn_tta, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.drug_embed_size = drug_embed_size

        self.transformer_drug = transformer(drug_embed_size, transf_dropout_rate, transf_n_layer, transf_intermediate_size, transf_n_att_heads, transf_att_prob_dropout, transf_hidden_dropout)

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map   

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)		   

        # ngenes, gene_dim are the number of all genes	
        self.gene_dim = ngene			   
        self.drug_dim = ndrug
        
        self.gamma = gamma
        # self.gene_embed_hid_dim = gene_embed_hid_dim
        self.gene_embed_dim = gene_embed_dim
        self.topN_else_mask = topN_else_mask
        # self.embed_dropout = embed_dropout
        
        self.gene_exp_vector_embed()
        # self.gene_embed_layer()
        self.lookup_table_embed()

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs	
        # self.construct_NN_drug()

        # add modules for final layer
        final_input_size = num_hiddens_genotype + drug_embed_size
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final,1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    
    def rbf_kernel(self, centers, gamma):
        vect = torch.zeros(centers.shape[0], centers.shape[1], centers.shape[1])
        # print(vect)
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


    # def gene_embed_layer(self):
    #     self.add_module('gene_embedding_layer', EmbedNet(self.gene_dim, self.gene_embed_hid_dim, self.gene_embed_dim, self.embed_dropout))
    
    
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
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1', nn.Linear(self.gene_embed_dim,1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(term_hidden,1))

            dG.remove_nodes_from(leaves)


    # definition of forward function
    def forward(self, cell_feature, drug_feature, drug_idx, table):

        # lookup table creation
        # table: lookup table을 만들기 위한 matrix
        lookup_table = self.create_lookup_table(table, drug_idx).cuda(cell_feature.device)
        # print('lookup_table shape:', lookup_table.shape)
        # print('lookup_table:', lookup_table)
        if self.topN_else_mask:
            gene_input = torch.mul(cell_feature, lookup_table)

            # gene expression scalar value vector embedding
            gene_input_rbf = self.rbf_kernel(gene_input, self.gamma)
            gene_input_vect = self._modules['RBF_embedding_layer'](gene_input_rbf.cuda(cell_feature.device))

            gene_input_final_permute = gene_input_vect.permute(0, 2, 1)
        else:
            lookup_table_vect = self._modules['lookup_table_embedding_layer'](lookup_table)
        
            # gene expression scalar value vector embedding
            gene_input_rbf = self.rbf_kernel(cell_feature, self.gamma)
            # print('through rbf kernel:', gene_input_rbf.shape)   
            gene_input_vect = self._modules['RBF_embedding_layer'](gene_input_rbf.cuda(cell_feature.device))

            # addition of two values to finally create final input
            gene_input_final = torch.add(gene_input_vect, lookup_table_vect)
            # print('final input shape:', gene_input_final.shape)
        
            gene_input_final_permute = gene_input_final.permute(0, 2, 1)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input_final_permute)
#             print(f'{term}_direct_gene_layer: {term_gene_out_map[term].shape}')
        
        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child].permute(0, 2, 1))

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

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
#                 print(f'{term}_batchnorm_layer: {term_NN_out_map[term].shape}')
                
                aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term])).squeeze()
#                 print(f'{term}_aux_layer_1_out: {aux_layer1_out.shape}')
                
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)
#                 print(f'{term}_aux_linear_layer2: {aux_out_map[term].shape}')

        # define forward function for drug dcell #################################################
        drug_out = drug_feature

        drug_out = self.transformer_drug(drug_out)
        
        # connect two neural networks at the top #################################################
        root_map = term_NN_out_map[self.root].mean(dim=2)
        final_input = torch.cat((root_map, drug_out), 1)
#         print('concatenated shape:', final_input.shape)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map
