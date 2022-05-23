# Embedding to Network Alignment (ETNA) algorithm
# With this class you can specify your own ETNA model and train it.
#
# ETNA builds off of SDNE (https://github.com/CyrilZhao-sudo/SDNE)
# using different proximity functions to calculate embedding.
# ETNA uses the deep walk approximation from
# NetMF (https://arxiv.org/abs/1710.02971) as input to consider
# the global structure of the graph and use cross training function
# to align two embeddings

from collections import defaultdict

import algorithms.helper as helper
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class Trainer(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass


class EmbeddingModel(torch.nn.Module):
    """network embedding model

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

    Attributes
    ----------
    encoder : Sequential
        encoder for the model

    decoder : Sequential
        decoder for the model

    loss_1st_f : BCEWithLogitsLoss
        loss function for the first order loss

    """

    def __init__(self, input_dim, hidden_layers=None, depth=2, device="cpu", bias=True):
        super(EmbeddingModel, self).__init__()
        self.device = device

        encoder = []
        dims = [input_dim] + hidden_layers
        for i in range(len(dims) - 1):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            encoder.append(nn.BatchNorm1d(dims[i + 1])),
            encoder.append(nn.LeakyReLU(0.1))
        self.encoder = torch.nn.Sequential(*encoder)

        self.encoder.apply(self.init_weights)

        decoder = []
        for i in range(len(dims) - 1, 1, -1):
            decoder.append(nn.Linear(
                dims[i], dims[i - 1]))
            decoder.append(nn.BatchNorm1d(dims[i - 1]))
            decoder.append(nn.LeakyReLU(0.1))
        decoder.append(torch.nn.Linear(dims[1], dims[0]))
        self.decoder = torch.nn.Sequential(*decoder)

        self.decoder.apply(self.init_weights)
        self.loss_1st_f = nn.BCELoss(reduction='none')
        # self.loss_1st_f = nn.BCEWithLogitsLoss(reduction='none')

    def init_weights(self, m):
        '''Initialize weight

        Parameters
        ----------
        m : torch.Linear
        '''
        if type(m) == nn.Linear:
            torch.manual_seed(0)
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, X, A):
        '''Forward propagation of the model

        Parameters
        ----------
        X : torch.Tensor
            Input matrix
            Shape `(batch_size, input_dim)`

        A : torch.Tensor
            Adjacency matrix
            Shape `(input_dim, input_dim)`

        Returns
        -------
        loss_2nd : torch.Tensor
            second order proximity loss (global similarity)

        loss_1st : torch.Tensor
            first order proximity loss (local similarity)

        loss_norm : torch.Tensor
            embedding norm loss
        '''
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
        '''Get encoder weigth parameter

        Returns
        -------
        params : torch.Tensor
            encoder weight parameters

        '''

        params = []
        params = params + list(self.encoder.parameters())

        return params

    def reg(self):
        '''Get l2 regularization loss

        Returns
        -------
        reg_loss : torch.Tensor
            l2 regularization loss

        '''
        reg_loss = 0
        for n, w in self.encoder.named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        for n, w in self.decoder.named_parameters():
            if n.split('.')[1] == 'weight':
                l2_reg = torch.norm(w, p=2)
                reg_loss += l2_reg
        reg_loss = reg_loss

        return reg_loss


class EmbeddingTrainer(Trainer):
    """Trainer for embedding model

    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        The input graph

    embedding_model : EmbeddingModel
        The embedding Model

    alpha : float
        The hyperparameter for the weight of 2nd order loss

    gamma : float
        the hyperparameter for the weight of 1st order loss

    reg : float
        the hyperparameter for the weight of regularization

    wnorm : float
        the hyperparameter for the weight of weight normalization

    window : int
        the hyperparameter for the weight of window size

    device : str
        the device use for training

    precal : bool
        whether to use precalculated matrix information

    matrices : tuple
        tuple of precalculated matrix (adjacency, deep walk, etc)


    Attributes
    ----------
    graph : networkx.classes.graph.Graph
        input graph

    node_size : int
        number of nodes in the graph

    edge_size : int
        number of edges in the graph

    device : str
        device name for training

    emb_model : EmbeddingModel
        The embedding Model

    optimizer : torch.optim.Adam
        optimizer for training

    scheduler : torch.optim.lr_scheduler.StepLR
        scheduler for optimization

    alpha : float
        hyper-parameter for 2nd order proximity loss

    embeddings : numpy.ndarray
        embedding generated from the encoder

    gamma : float
        hyper-parameter for 1st order proximity loss

    reg : float
        hyper-parameter for l2 regularization loss

    wnorm : float
        hyper-parameter for embedding norm loss

    batch_size : int
        the number of batch size

    adjacency_matrix : torch.tensor
        adjacency model of the graph

    dw_matrix : torch.tensor
        deep walk approximation matrix of the graph

    normalized_matrix : torch.tensor
        normalized deep walk approximation matrix

    """

    def __init__(self, graph, embedding_model, alpha=1.0, gamma=1.0, reg=1.0, wnorm=1.0,
                 window=10, device="cpu", precal=False, matrices=()):
        super().__init__()
        self.graph = graph
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.device = device
        self.emb_model = embedding_model
        self.optimizer = torch.optim.Adam(
            self.emb_model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=1)
        self.embeddings = np.array([])
        self.gamma = gamma
        self.alpha = alpha
        self.reg = reg
        self.wnorm = wnorm
        self.batch_size = 1

        if precal:
            self.adjacency_matrix = torch.from_numpy(matrices[0]).float()
            self.dw_matrix = torch.from_numpy(matrices[1]).float()
            self.normalized_matrix = torch.from_numpy(matrices[2]).float()
        else:
            adjacency_matrix = nx.adjacency_matrix(graph)
            dw_matrix = helper.direct_compute_deepwalk_matrix(
                adjacency_matrix, window).toarray()

            norms = np.linalg.norm(dw_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_matrix = dw_matrix / norms

            self.adjacency_matrix = torch.from_numpy(
                adjacency_matrix.toarray()).float()
            self.dw_matrix = torch.from_numpy(dw_matrix).float()
            self.normalized_matrix = torch.from_numpy(
                normalized_matrix).float()

    def fit(self, batch_size=64, epochs=1, initial_epoch=0, seed=None, verbose=1):
        '''Train the model
        Parameters
        ----------
            batch_size : int
                the number of batch size

            epochs : int
                the number of training epoch

            initial_epoch : int
                the number of epochs to start with

            verbose : int
                whether to print loss function detail

        '''
        self.emb_model.train()
        num_samples = self.node_size
        self.batch_size = batch_size

        steps_per_epoch = (self.node_size - 1) // batch_size
        indexes = np.arange(self.adjacency_matrix.shape[0])
        if seed:
            np.random.seed(seed)
        np.random.shuffle(indexes)
        for epoch in range(initial_epoch, epochs):
            loss_epoch = 0
            for i in range(steps_per_epoch):
                idx = np.arange(i * batch_size, min((i + 1)
                                                    * batch_size, self.node_size))
                idx = indexes[idx]
                X_train = self.normalized_matrix[idx, :].to(self.device)
                self.adjacency_matrix[idx, :].to(self.device)
                A_train = self.adjacency_matrix[idx, :][:, idx].to(self.device)
                self.optimizer.zero_grad()
                loss_2nd, loss_1st, loss_norm = \
                    self.emb_model(X_train, A_train)
                reg_loss = self.emb_model.reg()
                loss = self.gamma * loss_1st + self.alpha * loss_2nd +\
                    self.reg * reg_loss + self.wnorm * loss_norm
                loss_epoch += loss.item()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            if verbose > 0:
                print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(
                    epoch + 1, round(loss_epoch / num_samples, 4), epoch + 1, epochs))

    def get_embeddings(self):
        '''return the embedding

        Returns
        -------
        embeddins : numpy.ndarray
            embedding generated from the encoder

        '''
        self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        '''generate the embedding from encoder

        '''
        self.emb_model.eval()
        with torch.no_grad():
            self.emb_model.eval()
            embedding = np.array([])
            steps_per_epoch = (self.node_size - 1) // self.batch_size + 1
            for i in range(steps_per_epoch):
                idx = np.arange(i * self.batch_size,
                                min((i + 1) * self.batch_size, self.node_size))

                X_train = self.normalized_matrix[idx, :].to(self.device)
                embed = self.emb_model.encoder(X_train).cpu().detach().numpy()
                if len(embedding) == 0:
                    embedding = embed
                else:
                    embedding = np.concatenate((embedding, embed))
        self.embeddings = embedding


class ETNA(Trainer):
    '''ETNA model trainer

    Parameters
    ----------
    g1 : networkx.classes.graph.Graph
        The input graph for species 1

    g2 : networkx.classes.graph.Graph
        The input graph for species 2

    orthologs : set
        The set of orthologous protein pairs

    g1_hidden_layers : list
        speices 1 embedding model hidden layers

    g2_hidden_layers : list
        speices 2 embedding model hidden layers

    alpha1 : float
        The hyperparameter for the weight of 2nd order loss of
        speices 1 embedding model

    gamma1 : float
        the hyperparameter for the weight of 1st order loss of
        speices 1 embedding model

    reg1 : float
        the hyperparameter for the weight of regularization of
        speices 1 embedding model

    wnorm1 : float
        the hyperparameter for the weight of weight normalization of
        speices 1 embedding model

    window1 : int
        the hyperparameter for the weight of window size of
        speices 1 embedding model

    alpha2 : float
        The hyperparameter for the weight of 2nd order loss of
        speices 2 embedding model

    gamma2 : float
        the hyperparameter for the weight of 1st order loss of
        speices 2 embedding model

    reg2 : float
        the hyperparameter for the weight of regularization of
        speices 2 embedding model

    wnorm2 : float
        the hyperparameter for the weight of weight normalization of
        speices 2 embedding model

    window2 : int
        the hyperparameter for the weight of window size of
        speices 2 embedding model

    psi : float
        the hyperparameter for cross training

    device : str
        the device use for training

    precal : bool
        whether to use precalculated matrix information

    g1_matrices : tuple
        tuple of precalculated matrix (adjacency, deep walk, etc) of
        speices 1

    g2_matrices : tuple
        tuple of precalculated matrix (adjacency, deep walk, etc) of
        speices 2

    Attributes
    ----------
    emb_model1 : EmbeddingModel
        species 1 embedding model

    emb_model2 : EmbeddingModel
        species 2 embedding model

    emb_trainer1 : EmbeddingTrainer
        species 1 embedding trainer

    emb_trainer2 : EmbeddingTrainer
        species 2 embedding trainer

    orthologs : set
        The set of orthologous protein pairs

    optimizer_align : torch.optim.Adam
        optimizer for alignment

    scheduler_align : torch.optim.lr_scheduler.StepLR
        scheduler for alignment

    device : str
        the device to train

    psi : float
        the hyperparameter for cross training
    '''

    def __init__(self, g1, g2, orthologs, g1_hidden_layers=[1024, 128],
                 g2_hidden_layers=[1024, 128], alpha1=1.0, gamma1=1.0, reg1=1.0,
                 wnorm1=1.0, window1=10, alpha2=1.0, gamma2=1.0, reg2=1.0,
                 wnorm2=1.0, window2=10, psi=1.0, device="cpu", precal=False,
                 g1_matrices=(), g2_matrices=()):
        super().__init__()

        self.emb_model1 = EmbeddingModel(len(g1.nodes), hidden_layers=[1024, 128],
                                         device=device).to(device)
        self.emb_model2 = EmbeddingModel(len(g2.nodes), hidden_layers=[1024, 128],
                                         device=device).to(device)
        self.emb_trainer1 = EmbeddingTrainer(g1, self.emb_model1, alpha=alpha1,
                                             gamma=gamma1, reg=reg1, wnorm=wnorm1,
                                             window=window1, device=device,
                                             precal=precal, matrices=g1_matrices)
        self.emb_trainer2 = EmbeddingTrainer(g2, self.emb_model2, alpha=alpha2,
                                             gamma=gamma2, reg=reg2, wnorm=wnorm2,
                                             window=window2, device=device,
                                             precal=precal, matrices=g2_matrices)
        self.orthologs = orthologs

        self.optimizer_align = torch.optim.Adam(self.emb_model1.encoders_parameters() +
                                                self.emb_model2.encoders_parameters())
        self.scheduler_align = torch.optim.lr_scheduler.StepLR(self.optimizer_align,
                                                               step_size=1,
                                                               gamma=1)
        self.device = device
        self.psi = psi

    def fit(self, emb_epoch=1, align_epoch=1, total_epoch=10, batch_size=128, verbose=0):
        '''Train the model

        Parameters
        ----------
        emb_epoch : int
            the number of embedding epoch

        align_epoch : int
            the number of alignment epoch

        total_epoch : int
            the number of ETNA epoch (embedding + alignment)

        batch_size : int
            the number of batch size

        verbose : int
            whether print detailed loss
        '''
        for i in range(total_epoch):
            self.emb_trainer1.fit(batch_size=batch_size, epochs=emb_epoch,
                                  seed=i, verbose=0)
            self.emb_trainer2.fit(batch_size=batch_size, epochs=emb_epoch,
                                  seed=i, verbose=0)

            for k in range(align_epoch):
                self.cross_training(seed=i)

    def cross_training(self, seed=None):
        '''cross training alignment for two embeddings

        Parameters
        ----------
        seed : int
            random seed for shuffling ortholog
        '''
        self.emb_trainer1.emb_model.train()
        self.emb_trainer2.emb_model.train()
        orthologs = list(self.orthologs)


        if seed:
            np.random.seed(seed)
        np.random.shuffle(orthologs)
        anchor_x = np.array([x for x, y in orthologs])
        anchor_y = np.array([y for x, y in orthologs])

        steps_per_epoch = (len(orthologs) - 1) // 128
        loss = 0
        for i in range(steps_per_epoch):
            idx = np.arange(i * 128, min((i + 1) * 128, len(orthologs)))
            epoch_x = anchor_x[idx]
            epoch_y = anchor_y[idx]


            self.optimizer_align.zero_grad()
            X1 = self.emb_trainer1.normalized_matrix[epoch_x].to(self.device)
            X2 = self.emb_trainer2.normalized_matrix[epoch_y].to(self.device)

            emb1 = self.emb_trainer1.emb_model.encoder(X1)
            recon1 = self.emb_trainer2.emb_model.decoder(emb1)

            loss1 = F.binary_cross_entropy_with_logits(
                recon1, X2, reduction='none')
            loss1 = torch.mean(torch.sum(loss1, dim=1) )

            emb2 = self.emb_trainer2.emb_model.encoder(X2)
            recon2 = self.emb_trainer1.emb_model.decoder(emb2)

            loss2 = F.binary_cross_entropy_with_logits(
                recon2, X1, reduction='none')
            loss2 = torch.mean(torch.sum(loss2, dim=1) )

            loss = self.psi * (loss1 + loss2)

            loss.backward()
            self.optimizer_align.step()
        self.scheduler_align.step()

    def get_score_matrix(self):
        '''calculate score matrix

        Returns
        ----------
        score_matrix : numpy.ndarray
            score matrix calculated from embeddings
        '''
        emb1 = self.emb_trainer1.get_embeddings()
        emb2 = self.emb_trainer2.get_embeddings()
        score_matrix = metrics.pairwise.cosine_similarity(emb1, emb2)
        return score_matrix
