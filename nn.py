import torch
import torch.nn as nn
import numpy as np
import pdb

def generate_x(n, d, epsilon):
    x1 = np.random.rand(d)
    x1[2:] = 0
    x1 = x1 / np.linalg.norm(x1)

    x2 = np.random.rand(d)
    x2[0] = 0
    x2[3:] = 0
    x2 = x2 / np.linalg.norm(x2)

    X = np.stack([x1, x2], axis = 0)
    idx = np.random.randint(0, 2, n - 2)
    X_random = X[idx] + np.random.rand(n - 2, d) * epsilon
    X_random = X_random /  np.linalg.norm(X_random, axis = -1)[:,None]
    return np.concatenate([X, X_random], axis = 0)

class neural_network():
    def __init__(self, d_in, d_out, train_epoch, lr, weight_decay, batch_size, dropout):
        self.d_in = d_in
        self.d_out = d_out
        self.train_epoch = train_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.reset_net()

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.net.parameters() if w.requires_grad)

    def reset_net(self):
        if not self.dropout:
            self.net = nn.Sequential(nn.Linear(self.d_in, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.d_out),
                                    )
        else:
            self.net = nn.Sequential(nn.Linear(self.d_in, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.Dropout(p = 0.5),
                                    nn.Linear(128, self.d_out),
                                    )
        self.net = self.net.cuda()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), self.lr,
            weight_decay=self.weight_decay)

    def SVD_X(self, X, epsilon_d):
        U, sigma, V_t = np.linalg.svd(X, full_matrices=False)
        d = min(X.shape[1], X.shape[0])
        rr = np.arange(1, d + 1)
        epsilon_zero = np.zeros(1)
        epsilon_rr = np.cumsum(sigma[::-1])[::-1][1:]
        epsilon_rr = np.concatenate([epsilon_rr, epsilon_zero])
        rr_filter = rr[np.where(epsilon_rr <=  epsilon_d)]
        d_r = min(rr_filter)
        X_r = (U @ np.diag(sigma))[:, :d_r]
        return X_r

    def compress_arms(self, input, epsilon_d):
        self.net.zero_grad()
        self.net.eval()
        input = torch.from_numpy(input).float().cuda()
        grad_list = []
        for i in range(input.shape[0]):
            output = self.net(input[[i]])
            output.mean().backward()
            grad = torch.cat([p.grad.flatten().detach() for p in self.net.parameters()])
            grad_list.append(grad)
        grad_array = torch.stack(grad_list, dim = 0)
        grad_array = grad_array / self.approximator_dim
        # grad_array = grad_array / 128
        print('approximator_dim:', self.approximator_dim, '\n')
        print(grad_array.cpu().data.float().numpy().shape)
        X_r = self.SVD_X(grad_array.cpu().data.float().numpy(), epsilon_d)
        return X_r

    def train_loop(self, arms, rewards):
        self.net.train()
        arms = torch.from_numpy(arms).float().cuda()
        rewards = torch.from_numpy(rewards).float().cuda()
        batch_size = int(rewards.shape[0] / 2)
        for i in range(self.train_epoch):
            arms_to_train = arms
            rewards_to_train = rewards
            pred_rewards = self.net(arms_to_train)
            loss = nn.MSELoss()(pred_rewards[:,0], rewards_to_train.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.net.eval()
