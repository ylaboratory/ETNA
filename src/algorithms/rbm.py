#https://github.com/eugenet12/pytorch-rbm-autoencoder
import torch

class RBM():
    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False, device='cpu'):
        """Initialize a Restricted Boltzmann Machine
        Parameters
        ----------
        visible_dim: int
            number of dimensions in visible (input) layer
        hidden_dim: int
            number of dimensions in hidden layer
        gaussian_hidden_distribution: bool
            whether to use a Gaussian distribution for the values of the hidden dimension instead of a Bernoulli
        """
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution
        self.device = device

        # intialize parameters
        self.W = torch.randn(visible_dim, hidden_dim).to(device) * 0.1
        self.h_bias = torch.zeros(hidden_dim).to(device)  # v --> h
        self.v_bias = torch.zeros(visible_dim).to(device)  # h --> v

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim).to(device)
        self.h_bias_momentum = torch.zeros(hidden_dim).to(device)  # v --> h
        self.v_bias_momentum = torch.zeros(visible_dim).to(device)  # h --> v

    def sample_h(self, v):
        """Get sample hidden values and activation probabilities
        Parameters
        ----------
        v: Tensor
            tensor of input from visible layer
        """
        activation = torch.mm(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, torch.normal(activation, torch.tensor([1]).to(self.device))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def sample_v(self, h):
        """Get visible activation probabilities
        Parameters
        ----------
        h: Tensor
            tensor of input from hidden
        """
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p

    def update_weights(self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size):
        """Learning step: update parameters 
        Uses contrastive divergence algorithm as described in
        Parameters
        ----------
        v0: Tensor
            initial visible state
        vk: Tensor
            final visible state
        ph0: Tensor
            hidden activation probabilities for v0
        phk: Tensor
            hidden activation probabilities for vk
        lr: float
            learning rate
        momentum_coef: float
            coefficient to use for momentum
        weight_decay: float
            coefficient to use for weight decay
        batch_size: int
            size of each batch
        """
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr*self.W_momentum/batch_size
        self.h_bias += lr*self.h_bias_momentum/batch_size
        self.v_bias += lr*self.v_bias_momentum/batch_size

        self.W -= self.W * weight_decay # L2 weight decay
        
        

def train_rbm(train_dl, visible_dim, hidden_dim, k, num_epochs, lr, use_gaussian=False, device='cpu'):
    """Create and train an RBM
    
    Uses a custom strategy to have 0.5 momentum before epoch 5 and 0.9 momentum after
    
    Parameters
    ----------
    train_dl: DataLoader
        training data loader
    visible_dim: int
        number of dimensions in visible (input) layer
    hidden_dim: int
        number of dimensions in hidden layer
    k: int
        number of iterations to run for Gibbs sampling (often 1 is used)
    num_epochs: int
        number of epochs to run for
    lr: float
        learning rate
    use_gaussian:
        whether to use a Gaussian distribution for the hidden state
    
    Returns
    -------
    RBM, Tensor, Tensor
        a trained RBM model, sample input tensor, reconstructed activation probabilities for sample input tensor
    """
    rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim, gaussian_hidden_distribution=use_gaussian, device=device)
    loss = torch.nn.MSELoss() # we will use MSE loss

    for epoch in range(num_epochs):
        train_loss = 0
        for i, data_list in enumerate(train_dl):
            sample_data = data_list[0].to(device)
            v0, pvk = sample_data, sample_data
            
            # Gibbs sampling
            for i in range(k):
                _, hk = rbm.sample_h(pvk)
                pvk = rbm.sample_v(hk)
            
            # compute ph0 and phk for updating weights
            ph0, _ = rbm.sample_h(v0)
            phk, _ = rbm.sample_h(pvk)
            
            
            # update weights
            rbm.update_weights(v0, pvk, ph0, phk, lr, 
                               momentum_coef=0.5 if epoch < 5 else 0.9, 
                               weight_decay=2e-4, 
                               batch_size=sample_data.shape[0])

            # track loss
            train_loss += loss(v0, pvk)
        
        # print training loss
        print(f"epoch {epoch}: {train_loss/len(train_dl)}")
    return rbm, v0, pvk
