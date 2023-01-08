import numpy as np
import logging
from linear_elim import linear_elim
from neural_elim import neural_elim
from kernel_elim import kernel_elim
from ada_linear_elim import ada_linear_elim
from ada_kernel_elim import ada_kernel_elim
import os
import sys
import functools
import pdb

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'linear_data_dir_binary')
data_pth = 'linear_data_dir_binary'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count], binary = True)
    return item_list[count]


def uniform_instance(n, d):
    while True:
        epsilon = 1e-5
        X_b = np.eye(d)
        factor = np.sqrt(d)
        x_1 = np.random.rand(d)
        x_1 = x_1 / np.sqrt((x_1**2).sum()) * np.sqrt(8/10)
        while True:
            x_2 = np.random.rand(d)
            x_2 = x_2 / np.sqrt((x_2**2).sum()) * np.sqrt(0.5)
            if x_1 @ x_2 < 0.405 and x_1 @ x_2 > 0.395:
                print('gap is:', x_1 @ x_2)
                X12 = np.concatenate([x_1.reshape(1, -1), x_2.reshape(1, -1)])
                if np.linalg.matrix_rank(X12) == 2:
                    print('rank:', np.linalg.matrix_rank(X12))
                    break
            else:
                print('unpass gap is:', x_1 @ x_2)
        n_split = int((n-2) / 2)
        X1_perturb = epsilon * np.random.randn(n_split)[:, None] * X_b[np.random.choice(d, n_split)] + x_1
        X2_perturb = epsilon * np.random.randn(n_split)[:, None] * X_b[np.random.choice(d, n_split)] + x_2
        X = np.concatenate([x_1.reshape(1, -1), x_2.reshape(1, -1), X1_perturb, X2_perturb])
        theta_star = x_1.reshape(-1, 1)
        Y = X @ theta_star
        if np.linalg.matrix_rank(X) == d:
            break
        else:
            print('do not pass the full rank filter')
    return X, Y, theta_star, epsilon


count = 50
delta = 0.05
sweep = [250, 500, 750, 1000]
factor = 10
pool_num = 5
d = 25 #dimensionality
epsilon_d = 1e-1

arguments = sys.argv[1:]

for n in sweep:

    np.random.seed(43)
    X_set = []
    theta_star_set = []
    Y_set = []
    for i in range(count):
        X, Y, theta_star, epsilon = uniform_instance(n, d)
        X_set.append(X)
        theta_star_set.append(theta_star)
        Y_set.append(Y)

    if 'kernel_elim' in arguments:
        np.random.seed(43)
        theta_norm = 0.5
        gamma = 1
        instance_list = [kernel_elim(X, theta_star, factor, \
                        delta, epsilon_d = epsilon_d, theta_norm = theta_norm,\
                        gamma = gamma) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        num_list = list(range(count))
        instance_list_new = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = True)
            instance_list_new.append(instance_list[num])
            if instance_list_new[-1].signal_break:
                instance_list_new[-1].success = 0
            print('Finished Kernel Elim Instance', len(instance_list_new))
            sample_complexity = np.array([instance.N for instance in instance_list_new])
            success = np.array([instance.success for instance in instance_list_new])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/kernel_' + str(n) + '.npy', [sample_complexity, success])

    # Adakernel_elim
    if 'ada_kernel_elim' in arguments:
        np.random.seed(43)
        theta_norm = 0.5
        gamma = 1
        instance_list = [ada_kernel_elim(X, theta_star, factor, \
                        delta, epsilon_d = epsilon_d, theta_norm = theta_norm, \
                        gamma = gamma) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        num_list = list(range(count))
        instance_list_new = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = True)
            instance_list_new.append(instance_list[num])
            if instance_list_new[-1].signal_break:
                instance_list_new[-1].success = 0
            print('Finished ada_kernel_elim Instance', len(instance_list_new))
            sample_complexity = np.array([instance.N for instance in instance_list_new])
            success = np.array([instance.success for instance in instance_list_new])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/ada_kernel_elim_' + str(n) + '.npy', [sample_complexity, success])

    # Linear_Elim
    if 'linear_elim' in arguments:
        np.random.seed(43)
        theta_norm = 0.5
        instance_list = [linear_elim(X, theta_star, factor, delta, \
                        epsilon_d = epsilon_d, theta_norm = theta_norm)\
                        for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        num_list = list(range(count))
        instance_list_new = []
        dr_list = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = True)
            instance_list_new.append(instance_list[num])
            if instance_list_new[-1].signal_break:
                instance_list_new[-1].success = 0
            print('Finished linear Instance', len(instance_list_new))
            sample_complexity = np.array([instance.N for instance in instance_list_new])
            success = np.array([instance.success for instance in instance_list_new])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        dr_list = [instance.dr_list for instance in instance_list_new]
        np.save(data_pth + '/lineardr_' + str(n) + '.npy', dr_list, 'dtype=object')
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/linear_' + str(n) + '.npy', [sample_complexity, success])

    # AdaLinear_Elim
    if 'ada_linear_elim' in arguments:
        np.random.seed(43)
        theta_norm = 0.5
        instance_list = [ada_linear_elim(X, theta_star, factor, delta, \
                        epsilon_d = epsilon_d, theta_norm = theta_norm)\
                        for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        num_list = list(range(count))
        instance_list_new = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = True)
            instance_list_new.append(instance_list[num])
            print('Finished ada_linear_elim Instance', len(instance_list_new))
            sample_complexity = np.array([instance.N for instance in instance_list_new])
            success = np.array([instance.success for instance in instance_list_new])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/ada_linear_elim_' + str(n) + '.npy', [sample_complexity, success])

    # Neural
    if 'neural_elim' in arguments:
        np.random.seed(43)
        instance_list = [neural_elim(X, theta_star, factor, \
                        delta, epsilon_k = epsilon_d, epsilon_d2 = 1e-3) \
                        for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        num_list = list(range(count))
        instance_list_new = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = True)
            instance_list_new.append(instance_list[num])
            print('Finished neural elim Instance', len(instance_list_new))
            success = np.array([instance.success for instance in instance_list_new])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/neural_' + str(n) + '.npy', [sample_complexity, success])
