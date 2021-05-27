
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


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

'''
joint SDNE model for multiple input graphs
'''

class SDNEJoint(torch.nn.Module):

    def __init__(self, input_dims, hidden_layers=None, device="cuda:1", sharing=True):
        '''
        Joint Structural Deep Network Embedding (SDNE Joint)
        :param input_dims: list of node size for input graphs
        :param hidden_layers: hidden layer node numbers for the autoencoder
        :param device: device that the model trained on
        :param sharing: whether sharing the inner layer between models for different graphs
        '''
        super(SDNEJoint, self).__init__()
        self.device = device

        self.encoders = []
        self.decoders = []
        num_params = len(hidden_layers)
        if sharing:
            encoder_sharing = torch.nn.Linear(
                hidden_layers[-2], hidden_layers[-1])
            decoder_sharing = torch.nn.Linear(
                hidden_layers[-1], hidden_layers[-2])
            num_params -= 1

        #initalize weight parameters based on input layer numbers
        #excpet the last layer, every layer of weight is followed by 
        #an ReLU activation layer
        for i in range(len(input_dims)):

            encoder = []
            layer_dims = [input_dims[i]] + hidden_layers
            for j in range(num_params):
                encoder.append(torch.nn.Linear(
                    layer_dims[j], layer_dims[j + 1]))
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
                decoder.append(torch.nn.Linear(
                    layer_dims[j], layer_dims[j - 1]))
                decoder.append(torch.nn.LeakyReLU(0.0))

            decoder.append(torch.nn.Linear(layer_dims[1], layer_dims[0]))
            self.decoder = torch.nn.Sequential(*decoder)
            self.decoder.apply(self.init_weights)
            self.decoders.append(self.decoder)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X, A, index, beta=10, k=0.05):
        '''
        get 1st and 2nd order loss for one graph
        :param X: input matrix for autoencoder
        :param A: adjacency matrix
        :param index: index of the graph
        :param beta: parameter for penalizing nonzero unit in 2nd order loss
        :param k: parameter for label smoothing
        '''
        Y = self.encoders[index](X)
        X_hat = self.decoders[index](Y)

        with torch.no_grad():
            X_label = (X - k) * X + k / 2
            A = (A - k) * A + k / 2

        beta_matrix = X * (beta - 1) + 1
        loss_2nd = F.binary_cross_entropy_with_logits(
            X_hat, X_label, reduction='none')
        loss_2nd = torch.mean(
            torch.sum(torch.mul(loss_2nd, beta_matrix), dim=1))

        S = torch.mm(Y, Y.t())
        loss_1st = F.binary_cross_entropy_with_logits(S, A)

        return loss_2nd, loss_1st

    def parameters(self, index):
        '''
        get parameter for one graph
        :param index: index of the graph
        '''
        params = []
        for para in self.encoders[index].parameters():
            params.append(para)
        for para in self.decoders[index].parameters():
            params.append(para)

        return params

    def encoders_parameters(self, index):
        '''
        get encoder parameter for one graph
        :param index: index of the graph
        '''
        params = []
        for para in self.encoders[index].parameters():
            params.append(para)

        return params

    def reg(self, index, p=2):
        '''
        get regularization loss for one graph
        :param index: index of the graph
        :param p: the order of regularization
        '''
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

    def __init__(self, graph, index, sdne_model, alpha=1, beta=10, gamma=1, reg=1, device="cuda:1"):
        '''
        Structural Deep Network Embedding (SDNE)
        :param graph: input graph
        :param index: the index of graph in the joint model
        :param sdne_model: joint model
        :param alpha: weight parameter for 2nd order loss
        :param beta: parameter for penalizing nonzero unit in 2nd order loss
        :param gamma: weight parameter for 1st order loss
        :param reg: weight parameter for regularization loss
        :param device: device that the model trained on
        '''
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.device = device
        self.sdne = sdne_model
        self.index = index
        self.optimizer = torch.optim.Adam(self.sdne.parameters(self.index))
        self.embeddings = {}
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.reg = reg

        adjacency_matrix = nx.adjacency_matrix(graph)
        laplace_matrix = nx.laplacian_matrix(graph)
        self.adjacency_matrix = torch.from_numpy(
            adjacency_matrix.toarray()).float()
        self.laplace_matrix = torch.from_numpy(
            laplace_matrix.toarray()).float()

    def fit(self, batch_size=64, epochs=1, initial_epoch=0, verbose=1):
        num_samples = self.node_size

        steps_per_epoch = (self.node_size - 1) // batch_size + 1
        for epoch in range(initial_epoch, epochs):
            loss_epoch = 0
            for i in range(steps_per_epoch):
                idx = np.arange(i * batch_size, min((i + 1)
                                                    * batch_size, self.node_size))
                X_train = self.adjacency_matrix[idx, :].to(self.device)
                A_train = self.adjacency_matrix[idx, :][:, idx].to(self.device)
                self.optimizer.zero_grad()
                loss_2nd, loss_1st = self.sdne(
                    X_train, A_train, self.index, beta=self.beta)
                reg_loss = self.sdne.reg(self.index)
                loss = self.gamma * loss_1st + self.alpha * loss_2nd + self.reg * reg_loss
                loss_epoch += loss.item()
                loss.backward()

                self.optimizer.step()

            if verbose > 0:
                print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                       epoch + 1, epochs))

    def get_embeddings(self):
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
                idx = np.arange(i * 128, min((i + 1) * 128, self.node_size))
                X_train = self.adjacency_matrix[idx, :].to(self.device)
                embed = self.sdne.encoders[self.index](
                    X_train).cpu().detach().numpy()
                if len(embedding) == 0:
                    embedding = embed
                else:
                    embedding = np.concatenate((embedding, embed))
        self.embeddings = embedding


def cross_training(models, orthologs, indexes, ortholog_matrices, optim, device="cuda:1", beta=10, k=0.05):
    '''
    cross training alignment for multiple graph's models
    :param models: SDNE model for different graphs
    :param orthologs: list of othologs set for different pairs of organisms
    :param indexes: list of pairs of indexes in the joint model
    :param ortholog_matrices: list of otholog matrix for different pairs of organisms
    :param optim: optimizer
    :param device: device that the model trained on
    :param beta: parameter for penalizing nonzero unit in the cross training loss
    :param k: parameter for label smoothing
    '''
    length = min([len(x) for x in orthologs])
    steps_per_epoch = (length - 1) // 128 + 1
    for i in range(steps_per_epoch):
        loss = 0
        for j in range(len(orthologs)):
            index1 = indexes[j][0]
            index2 = indexes[j][1]
            m1 = models[index1]
            m2 = models[index2]
            ortholog = list(orthologs[j])
            ortholog_matrix = ortholog_matrices[j]
            random.shuffle(ortholog)
            anchor_x = np.array([x for x, y in ortholog])
            anchor_y = np.array([y for x, y in ortholog])

            idx = np.arange(i * 128, min((i + 1) * 128, len(ortholog)))
            epoch_x = anchor_x[idx]
            epoch_y = anchor_y[idx]
            alignment_matrix = ortholog_matrix[
                epoch_x][:, epoch_y].float().to(device)

            optim.zero_grad()
            X1 = m1.adjacency_matrix[epoch_x].to(device)
            X2 = m2.adjacency_matrix[epoch_y].to(device)
            with torch.no_grad():
                label1 = (X2 - k) * X2 + k / 2
                label2 = (X1 - k) * X1 + k / 2
                alignment_labels = (alignment_matrix - k) * \
                    alignment_matrix + k / 2

            beta_matrix1 = X2 * (beta - 1) + 1
            emb1 = m1.sdne.encoders[index1](X1)
            recon1 = m2.sdne.decoders[index2](emb1)
            loss1 = F.binary_cross_entropy_with_logits(
                recon1, label1, reduction='none')
            loss1 = torch.mean(
                torch.sum(torch.mul(loss1, beta_matrix1), dim=1))
            loss += loss1

            beta_matrix2 = X1 * (beta - 1) + 1
            emb2 = m2.sdne.encoders[index2](X2)
            recon2 = m1.sdne.decoders[index1](emb2)
            loss2 = F.binary_cross_entropy_with_logits(
                recon2, label2, reduction='none')
            loss2 = torch.mean(
                torch.sum(torch.mul(loss2, beta_matrix2), dim=1))
            loss += loss2

            S = torch.mm(emb1, emb2.T)
            loss_align = F.binary_cross_entropy_with_logits(
                S, alignment_labels, reduction='none')
            loss_align = torch.sum(torch.mul(loss_align, alignment_matrix))
            loss += loss_align

        loss.backward()
        optim.step()
