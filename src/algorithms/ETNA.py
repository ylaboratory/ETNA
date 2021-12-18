# based on https://github.com/CyrilZhao-sudo/SDNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from collections import defaultdict

import algorithms.helper as helper
import algorithms.rbm as rbm


class GraphBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass


class ETNAModel(torch.nn.Module):
    """modified ETNA model 

    Parameters
    ----------
    input_dim : int
        The input dimension

    hidden_layers : list
        list of dimension of hidden layers

    depth : int
        number of attention layers

    device : str
        the device use for training

    bias : bool
        whether have bias for weight parameters

    rbm_init : bool
        whether to use rbm to do initialization

    rbms : RBM
        restrict boltzmann machine

    Attributes
    ----------
    encoder : Sequential
        encoder for the model

    decoder : Sequential
        decoder for the model

    loss_1st_f : BCEWithLogitsLoss
        loss function for the first order loss 

    """

    def __init__(self, input_dim, hidden_layers=None, depth=2, device="cpu", bias=True, rbm_init=False, rbms=None):
        super(ETNAModel, self).__init__()
        self.device = device

        # self.mean = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.var = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
        # torch.nn.init.xavier_uniform(self.mean.weight)
        # torch.nn.init.xavier_uniform(self.var.weight)
        # self.mean.bias.data.fill_(0.01)
        # self.var.bias.data.fill_(0.01)
        encoder = []
        dims = [input_dim]+hidden_layers
        for i in range(len(dims)-1):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            encoder.append(nn.BatchNorm1d(dims[i+1])),
            encoder.append(nn.LeakyReLU(0.1))
        self.encoder = torch.nn.Sequential(*encoder)

        if rbm_init is False:
            self.encoder.apply(self.init_weights)
        else:
            i = 0
            with torch.no_grad():
                for m in self.encoder:
                    if type(m) == nn.Linear:
                        m.weight.copy_(rbms[i].W.T.clone())
                        m.bias.copy_(rbms[i].h_bias.clone())
                        i += 1

        decoder = []
        for i in range(len(dims)-1, 1, -1):
            decoder.append(nn.Linear(
                dims[i], dims[i - 1]))
            decoder.append(nn.BatchNorm1d(dims[i-1]))
            decoder.append(nn.LeakyReLU(0.1))
        decoder.append(torch.nn.Linear(dims[1], dims[0]))
        self.decoder = torch.nn.Sequential(*decoder)
        
        if rbm_init is False:
            self.decoder.apply(self.init_weights)
        else:
            with torch.no_grad():
                for m in self.decoder:
                    if type(m) == nn.Linear:
                        i -= 1
                        m.weight.copy_(rbms[i].W.clone())
                        m.bias.copy_(rbms[i].v_bias.clone())

        self.loss_1st_f = nn.BCELoss(reduction='none')
        # self.loss_1st_f = nn.BCEWithLogitsLoss(reduction='none')

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.manual_seed(0)
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)                 # sampling epsilon
        z = mean + var * epsilon                          # reparameterization trick
        return z

    def forward(self, X, X_label, A, L):
        Z = self.encoder(X)

        X_hat = self.decoder(Z)

        loss_2nd = F.binary_cross_entropy_with_logits(
            X_hat, X, reduction='none')
        loss_2nd = torch.mean(
            torch.sum(loss_2nd, dim=1))

        S = torch.clamp(helper.pairwise_cosimilarity(Z, Z), min=0, max=1)

        loss_1st = self.loss_1st_f(S, A)
        loss_1st = torch.mean(torch.sum(loss_1st * A, dim=1))

        loss_norm = torch.mean(torch.norm(Z, dim=1))

        return loss_2nd, loss_1st, loss_norm

    def encoders_parameters(self):
        params = []
        params = params + list(self.encoder.parameters())
        # params = params + list(self.blocks.parameters())

        # for para in self.means.parameters():
        #    params.append(para)
        # for para in self.vars.parameters():
        #    params.append(para)

        return params

    def reg(self):
        reg_loss = 0
        for n, w in self.encoder.named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        # for n, w in self.mean.named_parameters():
        #    if n.split('.')[0] == 'weight':
        #        l2_reg = torch.norm(w, p=2)
        #        reg_loss += l2_reg
        # for n, w in self.var.named_parameters():
        #    if n.split('.')[0] == 'weight':
        #        l2_reg = torch.norm(w, p=2)
        #        reg_loss += l2_reg
        for n, w in self.decoder.named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        reg_loss = reg_loss * 0.01

        return reg_loss


class ETNATrainer(GraphBaseModel):
    """Trainer for ETNA

    Parameters
    ----------
    graph : networkx
        The input graph

    etna_model : ETNAModel
        The ETNA Model

    alpha : int
        The hyperparameter for the weight of 2nd order loss

    gamma : int
        the hyperparameter for the weight of 1st order loss

    reg : int
        the hyperparameter for the weight of regularization

    wnorm : int
        the hyperparameter for the weight of weight normalization

    window : int
        the hyperparameter for the weight of window size

    device : str
        the device use for training

    precal : bool
        whether to use precalculated matrix information

    matrices : tuple
        tuple of precalculated matrix (adjacnecy, deep walk, etc)


    Attributes
    ----------



    """

    def __init__(self, graph, etna_model, alpha=1, gamma=1, reg=1, wnorm=1,
                 window=10, device="cpu", precal=False, matrices=()):
        super().__init__()
        self.graph = graph
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.device = device
        self.etna = etna_model
        self.optimizer = torch.optim.Adam(self.etna.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=1)
        self.embeddings = {}
        self.gamma = gamma
        self.alpha = alpha
        self.reg = reg
        self.wnorm = wnorm

        if precal:
            self.adjacency_matrix = torch.from_numpy(matrices[0]).float()
            self.laplace_matrix = torch.from_numpy(matrices[1]).float()
            self.dw_matrix = torch.from_numpy(matrices[2]).float()
            self.normalized_matrix = torch.from_numpy(matrices[3]).float()
        else:
            adjacency_matrix = nx.adjacency_matrix(graph)

            laplace_matrix = nx.laplacian_matrix(graph)

            dw_matrix = helper.direct_compute_deepwalk_matrix(
                adjacency_matrix, window).toarray()

            norms = np.linalg.norm(dw_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_matrix = dw_matrix / norms

            self.adjacency_matrix = torch.from_numpy(
                adjacency_matrix.toarray()).float()
            self.laplace_matrix = torch.from_numpy(
                laplace_matrix.toarray()).float()
            self.dw_matrix = torch.from_numpy(dw_matrix).float()
            self.normalized_matrix = torch.from_numpy(
                normalized_matrix).float()

    def fit(self, batch_size=64, epochs=1, initial_epoch=0, verbose=1):
        self.etna.train()
        num_samples = self.node_size

        steps_per_epoch = (self.node_size - 1) // batch_size + 1
        indexes = np.arange(self.adjacency_matrix.shape[0])
        # np.random.seed(0)
        np.random.shuffle(indexes)
        for epoch in range(initial_epoch, epochs):
            loss_epoch = 0
            for i in range(steps_per_epoch):
                idx = np.arange(i * batch_size, min((i + 1)
                                                    * batch_size, self.node_size))
                idx = indexes[idx]
                X_train = self.normalized_matrix[idx, :].to(self.device)
                X_label = self.adjacency_matrix[idx, :].to(self.device)
                A_train = self.adjacency_matrix[idx, :][:, idx].to(self.device)
                L_train = self.laplace_matrix[idx][:, idx].to(self.device)
                self.optimizer.zero_grad()
                loss_2nd, loss_1st, loss_norm = \
                    self.etna(X_train, X_label, A_train,
                              L_train)
                reg_loss = self.etna.reg()
                loss = self.gamma * loss_1st + self.alpha * loss_2nd +\
                    self.reg * reg_loss + self.wnorm * loss_norm
                loss_epoch += loss.item()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            if verbose > 0:
                print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                       epoch + 1, epochs))

    def get_embeddings(self):
        # if not self.embeddings:
        self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        self.etna.eval()
        with torch.no_grad():
            self.etna.eval()
            embedding = np.array([])
            steps_per_epoch = (self.node_size - 1) // 128 + 1
            for i in range(steps_per_epoch):
                idx = np.arange(i * 128, min((i + 1) * 128, self.node_size))

                X_train = self.normalized_matrix[idx, :].to(self.device)
                embed = self.etna.encoder(X_train).cpu().detach().numpy()
                if len(embedding) == 0:
                    embedding = embed
                else:
                    embedding = np.concatenate((embedding, embed))
        self.embeddings = embedding


def cross_training(m1, m2, orthologs, optim, scheduler, device='cpu', psi=1):
    m1.etna.train()
    m2.etna.train()
    orthologs = list(orthologs)

    org1_ortholog_dict = defaultdict(int)
    org2_ortholog_dict = defaultdict(int)
    for i, j in orthologs:
        org1_ortholog_dict[i] += 1
        org2_ortholog_dict[j] += 1

    # np.random.seed(0)
    np.random.shuffle(orthologs)
    anchor_x = np.array([x for x, y in orthologs])
    anchor_y = np.array([y for x, y in orthologs])

    steps_per_epoch = (len(orthologs) - 1) // 128
    loss = 0
    for i in range(steps_per_epoch):
        idx = np.arange(i * 128, min((i + 1) * 128, len(orthologs)))
        epoch_x = anchor_x[idx]
        epoch_y = anchor_y[idx]

        org1_weight = torch.from_numpy(
            np.array([1 / org1_ortholog_dict[x] for x in epoch_x])).to(device)
        org2_weight = torch.from_numpy(
            np.array([1 / org2_ortholog_dict[x] for x in epoch_y])).to(device)

        optim.zero_grad()
        X1 = m1.normalized_matrix[epoch_x].to(device)
        X2 = m2.normalized_matrix[epoch_y].to(device)

        X1_label = m1.adjacency_matrix[epoch_x].to(device)
        X2_label = m2.adjacency_matrix[epoch_y].to(device)

        emb1 = m1.etna.encoder(X1)
        recon1 = m2.etna.decoder(emb1)

        loss1 = F.binary_cross_entropy_with_logits(
            recon1, X2, reduction='none')
        loss1 = torch.mean(torch.sum(loss1, dim=1) * org1_weight)

        emb2 = m2.etna.encoder(X2)
        recon2 = m1.etna.decoder(emb2)

        loss2 = F.binary_cross_entropy_with_logits(
            recon2, X1, reduction='none')
        loss2 = torch.mean(torch.sum(loss2, dim=1) * org2_weight)

        loss = psi * loss1 + psi * loss2

        loss.backward()
        optim.step()
    scheduler.step()
