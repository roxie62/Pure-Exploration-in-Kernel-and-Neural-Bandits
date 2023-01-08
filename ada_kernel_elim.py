import numpy as np
import itertools
import logging
import time
import pdb
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import KernelPCA
import math
from sklearn.kernel_approximation import RBFSampler

def build_Y_zeta(X_sample):
    k, d_r = X_sample.shape[0], X_sample.shape[1]
    idxs = np.zeros((k*k,2))
    Zhat = X_sample
    Y = np.zeros((k*k, d_r))
    rangeidx = np.array(list(range(k)))
    for i in range(k):
        idxs[k*i:k*(i+1),0] = rangeidx
        idxs[k*i:k*(i+1),1] = i
        Y[k*i:k*(i+1),:] = Zhat - Zhat[i,:]
    Yhat_sample = Y
    return Yhat_sample

def optimal_allocation_zeta(X):

    K = X.shape[0]
    design = np.ones(K)
    design /= design.sum()
    max_iter = 5000
    Yhat_sample = build_Y_zeta(X)
    for count in range(1, max_iter):
        A_inv = np.linalg.pinv(X.T@np.diag(design)@X)
        U,D,V = np.linalg.svd(A_inv)
        Ainvhalf = U@np.diag(np.sqrt(D))@V
        #shape of newY: n^2 * d
        newY = (Yhat_sample@Ainvhalf)**2
        rho = newY@np.ones((newY.shape[1], 1))
        idx = np.argmax(rho)
        y = Yhat_sample[idx, :, None]
        g = ((X@A_inv@y)*(X@A_inv@y)).flatten()
        g_idx = np.argmax(g)
        gamma = 2/(count+2)
        design_update = -gamma*design
        design_update[g_idx] += gamma
        relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
        design += design_update
        if relative < 0.01:
                break
    return rho[idx]

def test_effective_dimension(X_id_list, sigma_rr, rr, d_r, epsilon_d, eps):
    sigma_id = sigma_rr[d_r-1]
    X_id = X_id_list[d_r-1]
    g_approx_id = np.sqrt((1 + eps) * optimal_allocation_zeta(X_id))
    filter = ((16+8*g_approx_id) * sigma_id) <= epsilon_d
    # print(filter)
    return filter


def calculate_effective_dimension(X, eps, epsilon_d, theta_norm = 0.5):
    # calculate \tilde gamma_d
    U, sigma, V_t = np.linalg.svd(X, full_matrices=False)
    sigma_d = np.zeros(1)
    sigma_rr = np.cumsum(sigma[::-1])[::-1][1:]
    d =len(sigma)
    d_r = d
    rr = np.arange(1, d + 1)
    # note when d_r = 1: sigma_rr[1 - 1] = sigma[1:].sum(); d_r = d: sigma_rr[d - 1] = 0
    sigma_rr = np.concatenate([sigma_rr, sigma_d])
    sigma_rr = theta_norm * np.sqrt(sigma_rr)
    # approximate g(d, \zeta)
    g_approx = np.sqrt(4 * np.arange(1, d+1) * (1 + eps))
    filter = ((16+8*g_approx) * sigma_rr) <= epsilon_d
    d_r = np.min(rr[filter])
    approximate = True

    gamma_tilde_d = sigma_rr[d_r - 1]
    X_r = (U @ np.diag(np.sqrt(sigma)))[:, :d_r]
    return d_r, X_r, gamma_tilde_d

def kernel_mapping(X, gamma, gaussian = True):
    # X: n * d
    # K: n * n
    if gaussian:
        sq_dist = pdist(X, 'sqeuclidean')
        # gamma = 1
        mat_sq_dists = squareform(sq_dist)
        K = np.exp(-gamma * mat_sq_dists)
    return K


class ada_kernel_elim(object):
    def __init__(self, X, reward_func, factor, delta, epsilon_d = 0.1, theta_norm = 1, gamma = 1, Z = None):

        self.X = X
        self.X_sample = X
        self.X_original = X.copy()
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.epsilon_d = epsilon_d
        self.reward_func = reward_func
        self.theta_norm = theta_norm
        if type(self.reward_func) == np.ndarray:
            self.max_rwds = np.max(self.Z@reward_func)
            self.opt_arm = np.where((self.max_rwds - self.Z @ reward_func).reshape(-1) <= epsilon_d)[0]
        else:
            true_rewards = self.reward_func(np.arange(self.K), star = True)
            self.max_rwds = np.max(true_rewards)
            self.opt_arm = np.where((self.max_rwds - true_rewards).reshape(-1) <= epsilon_d)[0]
        self.delta = delta
        self.factor = factor
        self.gamma = gamma

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
            rewards = rewards +  np.random.randn(allocation.sum(), 1)
        return rewards

    def algorithm(self, seed, binary=True):

        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
        eps = 1/self.factor
        self.X_sample = kernel_mapping(self.X_sample, self.gamma)
        self.signal_break = False
        self.X_reduction = self.X_sample.copy()
        self.success = 0
        self.dr_list = []


        while len(self.active_arms) > 1:
            epsilon_eff = 4 * (2**(-self.phase_index))
            self.d_r, self.X_sample, self.gamma_tilde_d = calculate_effective_dimension(self.X_reduction, \
                                                            eps, epsilon_eff, theta_norm = self.theta_norm)

            if len(list(set(self.opt_arm) & set(self.active_arms))) == len(self.active_arms):
                break
            else:
                self.dr_list.append((self.d_r, self.phase_index))
                print(
                'd_r:', self.d_r, 'phase number', self.phase_index, 'success rate:', \
                len(list(set(self.opt_arm) & set(self.active_arms))) / len(self.active_arms) \
                )
                print('\n')

            self.delta_t = self.delta/(self.phase_index**2)
            self.build_Y()
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            n_min = 2*self.factor*support

            eps = 1/self.factor
            self.epsilon_k = 2 * self.gamma_tilde_d  + self.gamma_tilde_d * np.sqrt((1+eps) * rho)
            num_samples = max(np.ceil(((2**(-self.phase_index) - self.epsilon_k)** (-2)) *2 * rho*(1+eps)*np.log(self.K_Z**2/self.delta_t)), n_min).astype(int)
            # print('num_of_samples:', num_samples)
            allocation = self.rounding(design, num_samples)
            pulls = np.vstack([np.tile(self.X_sample[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            # print('shape of pulls:', pulls.shape)
            rewards = self.sample_rewards_from_allocation(allocation, binary)

            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = np.linalg.pinv(pulls.T@pulls)@pulls.T@rewards

            if self.N + num_samples >= 1e8:
                self.signal_break = True
                break

            self.drop_arms()
            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples
            # self.K_Z = len(self.active_arms)

            logging.debug('arm counts %s' % str(self.arm_counts))
            logging.info('active arms %s' % str(self.active_arms))


        del self.Yhat_sample
        del self.idxs
        del self.X_sample
        del self.Z
        self.opt_arms_located = list(set(self.opt_arm) & set(self.active_arms))
        self.success = len(self.opt_arms_located) / len(self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))


    def build_Y(self):

        k = len(self.active_arms)
        idxs = np.zeros((k*k,2))
        Zhat = self.X_sample[self.active_arms]
        Y = np.zeros((k*k, self.d_r))
        rangeidx = np.array(list(range(k)))

        for i in range(k):
            idxs[k*i:k*(i+1),0] = rangeidx
            idxs[k*i:k*(i+1),1] = i
            Y[k*i:k*(i+1),:] = Zhat - Zhat[i,:]

        self.idxs = idxs
        self.Yhat_sample = Y

    def optimal_allocation(self):

        design = np.ones(self.K)
        design /= design.sum()

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X_sample.T@np.diag(design)@self.X_sample)
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V

            newY = (self.Yhat_sample@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = self.Yhat_sample[idx, :, None]
            g = ((self.X_sample@A_inv@y)*(self.X_sample@A_inv@y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))

            design += design_update

            if count % 100 == 0:
                logging.debug('design status %s, %s, %s, %s' % (self.seed, count, relative, np.max(rho)))

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


    def drop_arms(self):

        active_arms = self.active_arms.copy()
        for arm_idx in active_arms:
            arm = self.X_sample[arm_idx, :, None]
            for arm_idx_prime in active_arms:
                if arm_idx == arm_idx_prime:
                    continue
                arm_prime = self.X_sample[arm_idx_prime, :, None]
                y = arm_prime - arm

                if self.epsilon_k + np.sqrt(np.abs(y.T@self.A_inv@y) * 2 * np.log(self.K_Z**2/self.delta_t)) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
