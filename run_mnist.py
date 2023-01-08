import numpy as np
import logging
import os
import sys
import functools
import pdb
from neural_elim import neural_elim
from kernel_elim import kernel_elim
from linear_elim import linear_elim
from sklearn.decomposition import PCA
import mnist


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# seed = 43
arguments = sys.argv[1:]
if len(sys.argv) == 3:
    seed = int(sys.argv[-1])
    data_dir = os.path.join(os.getcwd(), 'mnist_test_gaussian_{}'.format(seed))
    data_pth = 'mnist_test_gaussian_{}'.format(seed)
else:
    seed = 2
    data_dir = os.path.join(os.getcwd(), 'mnist_test_gaussian')
    data_pth = 'mnist_test_gaussian'



if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count], binary = False)
    return item_list[count]

count = 25

delta = 0.05
sweep = [100]
factor = 10
pool_num = 4
# noise is added when pulling arms (with two options: binary / gaussaiane)
# and thus set to 0 as initialization
Y_noise = 0
epsilon_d = 1e-1

x_train, t_train, x_test, t_test = mnist.load()
x_train = x_train * 1.0 / x_train.max()
d = 200

def get_mnist_instance(x_train, t_train, num_per_digit = 20):
    while True:
        X = []
        Y = []
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        shuffle_x_train, shuffle_t_train = x_train.copy(), t_train.copy()
        shuffle_x_train = shuffle_x_train[index]
        shuffle_t_train = shuffle_t_train[index]
        for i in range(10):
            X.append(shuffle_x_train[shuffle_t_train == i][:num_per_digit])
            if i == 7:
                ylabel = 1
            elif i == 1 or i == 9:
                ylabel = 0.8
            elif i == 2:
                ylabel = 0.8
            else:
                ylabel = 0.5
            Y.append(np.ones((num_per_digit,)) * ylabel)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        if np.linalg.matrix_rank(X) >= d:
            U, sigma, V_t = np.linalg.svd(X, full_matrices=False)
            X = (U @ np.diag(sigma))[:, :d]
            print('X shape', X.shape[0], X.shape[1])
            print(np.linalg.matrix_rank(X))
            break
    return X, Y

def reward_func(Y, std):
    # input is an index array
    def get_reward(idx, star = False):
        if star:
            return Y[idx].astype(np.float32)
        else:
            return Y[idx].astype(np.float32) + std * np.random.randn(idx.shape[0])
    return get_reward

np.random.seed(seed)
X_set = []
Y_set = []
theta_star_set = []
for i in range(count):
    X, Y = get_mnist_instance(x_train, t_train)
    print('count number:', i)
    X_set.append(X)
    Y_set.append(Y)

for n in sweep:

    if 'kernel_elim' in arguments:
        print(kernel_elim)
        np.random.seed(seed)

        theta_norm = 2.5 * 1e-3
        gamma = 9e-5
        instance_list = [kernel_elim(X, reward_func(Y, Y_noise), factor, \
                        delta, epsilon_d = epsilon_d, theta_norm = theta_norm, \
                        gamma = gamma) for X, Y in zip(X_set, Y_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        num_list = list(range(count))

        import multiprocess
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool_num = 1

        pool = multiprocess.Pool(pool_num)

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            instance_list.append(instance)
            print('Finished Kernel Elim Instance', len(instance_list))
            sample_complexity = np.array([instance.N for instance in instance_list])

            print('sample_complexity', sample_complexity.shape)
            success = np.array([instance.success for instance in instance_list])
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list]))
            opt_arms_located = np.mean(opt_arms_located)
            np.save(data_pth + '/kernel_temp' + str(n) + '.npy', [sample_complexity, success])

        sample_complexity = np.array([instance.N for instance in instance_list])
        success = np.array([instance.success for instance in instance_list])
        np.save(data_pth + '/kernel_' + str(n) + '.npy', [sample_complexity, success])

    # Linear_Elim
    if 'linear_elim' in arguments:
        print('linear_elim')
        np.random.seed(seed)
        theta_norm = 1e-5
        instance_list = [linear_elim(X, reward_func(Y, Y_noise), factor, delta, \
                        epsilon_d = epsilon_d, theta_norm = theta_norm)\
                        for X, Y in zip(X_set, Y_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        num_list = list(range(count))

        # import multiprocess
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        # num = 19
        # instance_list[num].algorithm(seed_list[num], binary = False)
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
        # for num in num_list:
            instance_list.append(instance)
            print('Finished linear Instance', len(instance_list))
            sample_complexity = np.array([instance.N for instance in instance_list])
            success = np.array([instance.success for instance in instance_list])
            success = np.mean(success)
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list]))
            opt_arms_located = np.mean(opt_arms_located)
            np.save(data_pth + '/linear_temp' + str(n) + '.npy', [sample_complexity, success])

        sample_complexity = np.array([instance.N for instance in instance_list])
        success = np.array([instance.success for instance in instance_list])
        np.save(data_pth + '/linear_' + str(n) + '.npy', [sample_complexity, success])

    # Neural
    if 'neural_elim' in arguments:
        print('neural_elim')
        np.random.seed(seed)
        instance_list = [neural_elim(X, reward_func(Y, Y_noise), factor, \
                        delta, epsilon_k = epsilon_d, epsilon_d2 = 1e-4, dropout = True) \
                        for X, Y in zip(X_set, Y_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        num_list = list(range(count))
        instance_list_new = []
        for num in num_list:
            instance_list[num].algorithm(seed_list[num], binary = False)
            instance_list_new.append(instance_list[num])
            print('Finished neural elim Instance', len(instance_list_new))
            sample_complexity = np.array([instance.N for instance in instance_list_new])
            success = np.array([instance.success for instance in instance_list_new])
            opt_arms_located = np.array(len([instance.opt_arms_located for instance in instance_list_new]))
            opt_arms_located = np.mean(opt_arms_located)
        dr_list = [instance.dr_list for instance in instance_list_new]
        np.save(data_pth + '/neuraldr_' + str(n) + '.npy', dr_list, 'dtype=object')
        sample_complexity = np.array([instance.N for instance in instance_list_new])
        success = np.array([instance.success for instance in instance_list_new])
        np.save(data_pth + '/neural_' + str(n) + '.npy', [sample_complexity, success])
