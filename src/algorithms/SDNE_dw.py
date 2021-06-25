import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import csgraph

class GraphBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass

def process_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


class SDNEJoint(torch.nn.Module):
    def __init__(self, input_dims, hidden_layers=None, beta=10, device="cuda:1", sharing=True):
        super(SDNEJoint, self).__init__()
        self.device = device
        self.beta = beta
        
        self.encoders = []
        self.decoders = []
        num_params = len(hidden_layers)
        if sharing:
            encoder_sharing = torch.nn.Linear(hidden_layers[-2], hidden_layers[-1])
            decoder_sharing = torch.nn.Linear(hidden_layers[-1], hidden_layers[-2])
            num_params -= 1
        for i in range(len(input_dims)):
            
            encoder = []
            layer_dims = [input_dims[i]]+hidden_layers
            for j in range(num_params):
                encoder.append(torch.nn.Linear(layer_dims[j], layer_dims[j+1]))
                encoder.append(torch.nn.LeakyReLU(0.0))
            if sharing:
                encoder.append(encoder_sharing)
                encoder.append(torch.nn.LeakyReLU(0.0))
            self.encoder = torch.nn.Sequential(*encoder)
            self.encoder.apply(self.init_weights)
            self.encoders.append(self.encoder)

            decoder = []
            if sharing:
                decoder.append(decoder_sharing)
                decoder.append(torch.nn.LeakyReLU(0.0))
            for j in range(num_params, 1, -1):
                decoder.append(torch.nn.Linear(layer_dims[j], layer_dims[j-1]))
                decoder.append(torch.nn.LeakyReLU(0.0))
            
            decoder.append(torch.nn.Linear(layer_dims[1], layer_dims[0]))
            self.decoder = torch.nn.Sequential(*decoder)
            self.decoder.apply(self.init_weights)
            self.decoders.append(self.decoder)
            
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, X, X_label, A, M, L, index):
        Y = self.encoders[index](X)
        X_hat = self.decoders[index](Y)
        
        beta_matrix = X_label*(self.beta-1)+1
        loss_2nd = torch.mean(torch.sum(torch.pow((X_label - X_hat) * beta_matrix, 2), dim=1))
        
        
        S = torch.mm(Y, Y.t())
        
        loss_1st =  (2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), L), Y)))
        loss_dw = torch.mean(torch.pow((S-M), 2))
        
        loss_norm = torch.mean(torch.norm(Y, dim=1))
        
        return loss_2nd, loss_1st, loss_dw, loss_norm
    
    def parameters(self, index):
        params = []
        for para in self.encoders[index].parameters():
            params.append(para)
        for para in self.decoders[index].parameters():
            params.append(para)
            
        return params
    
    def encoders_parameters(self, index):
        params = []
        for para in self.encoders[index].parameters():
            params.append(para)
        
        return params
        
    def reg(self, index):
        reg_loss = 0
        for n, w in self.encoders[index].named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        for n, w in self.decoders[index].named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        reg_loss = reg_loss * 0.01
        
        return reg_loss
    
class SDNE(GraphBaseModel):

    def __init__(self, graph, index, sdne_model, alpha=1, beta=10, gamma=0, reg=1, 
                 delta=1, wnorm=1, window=5, normalized=True, device="cuda:1", 
                 verbose=False):
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.device = device
        self.sdne = sdne_model
        self.index = index
        self.optimizer = torch.optim.Adam(self.sdne.parameters(self.index), )
        self.embeddings = {}
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.reg = reg
        self.wnorm = wnorm
        self.normalized = normalized
        self.verbose = verbose

        adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        laplace_matrix = nx.laplacian_matrix(graph)
        dw_matrix = self.generate_dw_embedding(window, adjacency_matrix)
        norms = np.linalg.norm(adjacency_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = adjacency_matrix/norms
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float()
        self.dw_matrix = torch.from_numpy(dw_matrix).float()
        self.normalized_matrix = torch.from_numpy(normalized_matrix).float()

    def fit(self, batch_size=64, epochs=1, initial_epoch=0, verbose=1):
        num_samples = self.node_size
        #self.sdne.to(self.device)
        
        steps_per_epoch = (self.node_size - 1) // batch_size + 1
        for epoch in range(initial_epoch, epochs):
            loss_epoch = 0
            for i in range(steps_per_epoch):
                idx = np.arange(i * batch_size, min((i+1) * batch_size, self.node_size))
                if self.normalized:
                    X_train = self.normalized_matrix[idx, :].to(self.device)
                else:
                    X_train = self.adjacency_matrix[idx, :].to(self.device)
                X_label = self.adjacency_matrix[idx, :].to(self.device)
                A_train = self.adjacency_matrix[idx, :][:,idx].to(self.device)
                M_train = self.dw_matrix[idx, :][:,idx].to(self.device)
                L_train = self.laplace_matrix[idx][:,idx].to(self.device)
                self.optimizer.zero_grad()
                loss_2nd, loss_1st, loss_dw, loss_norm = \
                self.sdne(X_train, X_label, A_train, M_train, L_train, self.index)
                reg_loss = self.sdne.reg(self.index)
                if i == 0 and verbose:
                    print(loss_2nd.item(), loss_1st.item(), loss_dw.item(), loss_norm.item())
                loss = self.gamma*loss_1st+self.alpha*loss_2nd+self.delta*loss_dw+\
                       self.reg*reg_loss+self.wnorm*loss_norm
                loss_epoch += loss.item()
                loss.backward()
                self.optimizer.step()


    def generate_dw_embedding(self, window, adj):
        n = len(self.graph.nodes)
        vol = np.sum(adj)
        lap, d_rt = csgraph.laplacian(adj, normed=True, return_diag=True) 
        X = sparse.identity(n) - lap
        S = np.zeros_like(X)
        X_power = sparse.identity(n)
        for i in range(window):
            X_power = X_power.dot(X)
            S += X_power        
        S *= vol / window 
        D_rt_inv = sparse.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T) 
        M = np.log(M.astype(float))
        M[M<0] = 0
        #M = 1/(1+np.exp(-M))
        return M
    
    def get_embeddings(self):
        #if not self.embeddings:
        self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.sdne.eval()
            embedding = np.array([])
            steps_per_epoch = (self.node_size - 1) // 128 + 1
            for i in range(steps_per_epoch):
                idx = np.arange(i * 128, min((i+1) * 128, self.node_size))
                if self.normalized:
                    X_train = self.normalized_matrix[idx, :].to(self.device)
                else:
                    X_train = self.adjacency_matrix[idx, :].to(self.device)
                embed = self.sdne.encoders[self.index](X_train).cpu().detach().numpy()
                if len(embedding) == 0:
                    embedding = embed
                else:
                    embedding = np.concatenate((embedding, embed))
            
        self.embeddings = embedding
        
def cross_training_als(m1, m2, orthologs, optim1, optim2, device="cuda:1"):
    orthologs = list(orthologs)
    random.shuffle(orthologs)
    anchor_x = np.array([x for x, y in orthologs])
    anchor_y = np.array([y for x, y in orthologs])
    
    
    steps_per_epoch = (len(orthologs) - 1) // 128 + 1
    loss_epoch = 0
    for i in range(steps_per_epoch):
        idx = np.arange(i * 128, min((i+1) * 128, len(orthologs)))
        epoch_x = anchor_x[idx]
        epoch_y = anchor_y[idx]
        X1 = m1.normalized_matrix[epoch_x].to(device)
        X2_label = m2.adjacency_matrix[epoch_y].to(device)
        emb1 = m1.sdne.encoders[0](X1)
        with torch.no_grad():
            emb2_nograd = torch.from_numpy(m2.get_embeddings()).float().to(device)
        
        optim1.zero_grad()
        beta_matrix1 = X2_label*(10-1)+1
        recon1 = m2.sdne.decoders[1](emb1)
        
        M2 = m2.dw_matrix[epoch_y].to(device)
        S1 = torch.mm(emb1, emb2_nograd.T)
        loss1_cross = torch.mean(torch.sum(torch.pow((recon1 - X2_label)*
                                               beta_matrix1, 2), dim=1))
        loss1_als = torch.mean(torch.pow((S1-M2), 2))
        loss1 = loss1_cross+loss1_als
        
        loss1.backward()
        optim1.step()
        
        
        X2 = m2.normalized_matrix[epoch_y].to(device)
        X1_label = m1.adjacency_matrix[epoch_x].to(device)
        emb2 = m2.sdne.encoders[1](X2)
        with torch.no_grad():
            emb1_nograd = torch.from_numpy(m1.get_embeddings()).float().to(device)
        
        optim2.zero_grad()
        beta_matrix2 = X1_label*(10-1)+1
        recon2 = m1.sdne.decoders[0](emb2)
        
        M1 = m1.dw_matrix[epoch_x].to(device)
        S2 = torch.mm(emb2, emb1_nograd.T)
        loss2_cross = torch.mean(torch.sum(torch.pow((recon2 - X1_label)*
                                               beta_matrix2, 2), dim=1))
        loss2_als = torch.mean(torch.pow((S2-M1), 2))
        loss2 = loss2_cross+loss2_als
        
        loss2.backward()
        optim2.step()


def cross_training(m1, m2, orthologs, ortholog_matrix, optim, device="cuda:1"):
    orthologs = list(orthologs)
    random.shuffle(orthologs)
    anchor_x = np.array([x for x, y in orthologs])
    anchor_y = np.array([y for x, y in orthologs])
    
    
    steps_per_epoch = (len(orthologs) - 1) // 128 + 1
    loss_epoch = 0
    for i in range(steps_per_epoch):
        idx = np.arange(i * 128, min((i+1) * 128, len(orthologs)))
        epoch_x = anchor_x[idx]
        epoch_y = anchor_y[idx]
        alignment_matrix = ortholog_matrix[epoch_x][:, epoch_y].float().to(device)

        optim.zero_grad()
        X1 = m1.normalized_matrix[epoch_x].to(device)
        X2 = m2.normalized_matrix[epoch_y].to(device)
        X1_label = m1.adjacency_matrix[epoch_x].to(device)
        X2_label = m2.adjacency_matrix[epoch_y].to(device)
        

        beta_matrix1 = X2*(10-1)+1
        emb1 = m1.sdne.encoders[0](X1)
        recon1 = m2.sdne.decoders[1](emb1)
        #loss1 = F.binary_cross_entropy_with_logits(recon1, label1, reduction='none')
        #loss1 = torch.mean(torch.sum(torch.mul(loss1, beta_matrix1), dim=1))
        loss1 = torch.mean(torch.sum(torch.pow((recon1 - X2_label) * beta_matrix1, 2), dim=1))
        
        beta_matrix2 = X1*(10-1)+1
        emb2 = m2.sdne.encoders[1](X2)
        recon2 = m1.sdne.decoders[0](emb2)
        #loss2 = F.binary_cross_entropy_with_logits(recon2, label2, reduction='none')
        #loss2 = torch.mean(torch.sum(torch.mul(loss2, beta_matrix2), dim=1))
        loss2 = torch.mean(torch.sum(torch.pow((recon2 - X1_label) * beta_matrix2, 2), dim=1))
        
        #S = torch.mm(emb1, emb2.T)
        #loss_align = F.binary_cross_entropy_with_logits(S, alignment_matrix, reduction='none')
        #loss_align = torch.sum(torch.mul(loss_align, alignment_matrix))

        loss_align = torch.mean(torch.sum(torch.pow((emb1 - emb2), 2), dim=1))

        loss = loss1 + loss2 #+ loss_align
        #print(loss1.item(), loss2.item(), loss_align.item())
        loss.backward()
        optim.step()
        
