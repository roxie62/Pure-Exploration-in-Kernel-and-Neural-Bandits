import numpy as np
import itertools
import logging
import pdb
from nn import neural_network
import torch

class neural_elim(object):
    def __init__(self, X, reward_func, factor, delta, epsilon_k = 1e-1, epsilon_d2 = 1e-2, dropout = False, Z=None):

        self.X = X
        self.X_original = X.copy()
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.epsilon_d2 = epsilon_d2
        self.reward_func = reward_func
        self.epsilon_k = epsilon_k
        self.dropout = dropout
        if type(self.reward_func) == np.ndarray:
            self.max_rwds = np.max(self.Z@reward_func)
            self.opt_arm = np.where((self.max_rwds - self.Z @ reward_func).reshape(-1) <= epsilon_k)[0]
        else:
            true_rewards = self.reward_func(np.arange(self.K), star = True)
            self.max_rwds = np.max(true_rewards)
            self.opt_arm = np.where((self.max_rwds - true_rewards).reshape(-1) <= epsilon_k)[0]
        self.delta = delta
        self.factor = factor

    def sample_rewards_from_allocation(self, allocation, binary):
        if type(self.reward_func) == np.ndarray:
            pulls_from_x = np.vstack([np.tile(self.X_original[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            rewards = pulls_from_x@self.reward_func
        else:
            index = np.where(allocation > 0)[0]
            rewards = []
            for ix in index:
                pulls = allocation[ix]
                rewards.append(self.reward_func((np.ones(pulls) * ix).astype(np.int32)))
            rewards = np.concatenate(rewards).reshape(-1, 1)
        if binary:
            rewards = np.clip(rewards, 0, 1)
            rewards = np.random.binomial(1, rewards, (allocation.sum(), 1))
        else:
            rewards = rewards + np.random.randn(allocation.sum(), 1)
        return rewards

    def algorithm(self, seed, binary=True):

        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1

        train_epoch = 6000
        lr = 1e-4
        weight_decay = 1e-4
        batch_size = self.X.shape[0]

        self.nn = neural_network(self.d, 1, train_epoch, lr, weight_decay, batch_size, self.dropout)
        self.signal_break = False
        self.success = 0
        self.active_arms = list(range(len(self.Z)))
        self.dr_list = []

        while len(self.active_arms) > 1:
            self.X_sample= self.nn.compress_arms(self.X, self.epsilon_d2)

            if len(list(set(self.opt_arm) & set(self.active_arms))) == len(self.active_arms):
                break
            else:
                self.dr_list.append((self.X_sample.shape[1], self.phase_index))
                print('self.phase_index', self.phase_index, '- success rate:', len(list(set(self.opt_arm) & set(self.active_arms))) / len(self.active_arms), \
                'd_r is', self.X_sample.shape[1])


            self.delta_t = self.delta/(8 * self.phase_index**2)

            self.build_Y()
            design, rho = self.optimal_allocation()

            support = np.sum((design > 0).astype(int))
            n_min = 2*self.factor*support
            eps = 1/self.factor
            num_samples = max(np.ceil( (2**(2*self.phase_index)) *rho*(1+eps)*np.log(self.K_Z**2/self.delta_t)), n_min).astype(int)
            if self.N + num_samples >= 1e8:
                self.signal_break = True
                break
            # print('num_samples:', num_samples)
            allocation = self.rounding(design, num_samples)

            rewards = self.sample_rewards_from_allocation(allocation, binary)
            pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])

            if torch.cuda.device_count() > 0:
                with torch.no_grad():
                    pulls = torch.from_numpy(pulls).cuda()
                    pulls = pulls.float()
                    self.A_inv = torch.pinverse(torch.matmul(pulls.T, pulls))
                    pulls = pulls.cpu().data.numpy()
                    self.A_inv = self.A_inv.cpu().data.numpy()
            else:
                self.A_inv = np.linalg.pinv(pulls.T@pulls)

            self.nn.train_loop(pulls, rewards)

            self.drop_arms(self.epsilon_k)
            self.arm_counts += allocation
            self.N += num_samples
            self.phase_index += 1

        del self.Yhat
        del self.idxs
        del self.X
        del self.Z
        self.opt_arms_located = list(set(self.opt_arm) & set(self.active_arms))
        self.success = len(self.opt_arms_located) / len(self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))

    def build_Y(self):

        k = len(self.active_arms)
        idxs = np.zeros((k*k,2))
        Zhat = self.X_sample[self.active_arms]
        self.d_r = Zhat.shape[1]
        Y = np.zeros((k*k, self.d_r))
        rangeidx = np.array(list(range(k)))

        for i in range(k):
            idxs[k*i:k*(i+1),0] = rangeidx
            idxs[k*i:k*(i+1),1] = i
            Y[k*i:k*(i+1),:] = Zhat - Zhat[i,:]

        self.Yhat = Y
        self.idxs = idxs


    def optimal_allocation(self):

        design = np.ones(self.K)
        design /= design.sum()

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X_sample.T@np.diag(design)@self.X_sample)
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V
            newY = (self.Yhat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X_sample@A_inv@y)*(self.X_sample@A_inv@y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))

            design += design_update

            if relative < 0.01:
                 break

        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0

        return design, np.max(rho)


    def rounding(self, design, num_samples):

        num_support = (design > 0).sum()
        support_idx = np.where(design>0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - .5*num_support)*support)

        while n_round.sum()-num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round/support)
                n_round[idx] += 1
            else:
                idx = np.argmax((n_round-1)/support)
                n_round[idx] -= 1

        allocation = np.zeros(len(design))
        allocation[support_idx] = n_round

        return allocation.astype(int)


    def drop_arms(self, epsilon_k):
        scaling_factor = 1e5
        active_arms = self.active_arms.copy()
        input_arms = self.X[active_arms].copy()
        self.nn.net.eval()
        output = self.nn.net(torch.from_numpy(input_arms).float().cuda())[:,0].cpu()

        output_diff  = (output[:,None] - output[None,:]).data.numpy() * scaling_factor
        # N * p
        output_diff = -1 * output_diff.min(axis = 1)
        if np.unique(output_diff).shape[0] == 1:
            invalid_mask = np.zeros(0)
        else:
            invalid_mask = output_diff >= ((2**(-self.phase_index)/8 + 3/8*epsilon_k) * scaling_factor)
        if invalid_mask.sum() > 0:
            for arm_idx in np.where(invalid_mask)[0]:
                self.active_arms.remove(active_arms[arm_idx])
