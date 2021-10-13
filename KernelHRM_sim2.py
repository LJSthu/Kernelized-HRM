import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
from tqdm import tqdm
from torch.autograd import grad
import copy
from torch import nn
import argparse
from sklearn.utils.extmath import randomized_svd
import random
import torch.optim as optim
import math
import torch.nn.functional as F
from EIIL import LearnedEnvInvariantRiskMinimization

np.set_printoptions(precision=4)

from multiprocessing import cpu_count
import os
cpu_num = 15
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def data_generation(n1, n2, ps, pvb, pv, r, scramble):
    S = np.random.normal(0, 2, [n1, ps])
    V = np.random.normal(0, 2, [n1, pvb + pv])

    Z = np.random.normal(0, 1, [n1, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]

    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0/2

    noise = np.random.normal(0, 1.0, [n1, 1])

    Y = np.dot(S, beta) + noise + 5 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    index_pre = np.ones([n1, 1], dtype=bool)
    for i in range(pvb):
        D = np.abs(V[:, pv + i:pv + i + 1] * sign(r) - Y)
        pro = np.power(np.abs(r), -D * 5)
        selection_bias = np.random.random([n1, 1])
        index_pre = index_pre & (
                    selection_bias < pro)
    index = np.where(index_pre == True)
    S_re = S[index[0], :]
    V_re = V[index[0], :]
    Y_re = Y[index[0]]
    n, p = S_re.shape
    index_s = np.random.permutation(n)

    X_re = np.hstack((S_re, V_re))
    beta_X = np.vstack((beta, np.zeros((pv + pvb, 1))))

    X = torch.from_numpy(X_re[index_s[0:n2], :]).float()
    y =  torch.from_numpy(Y_re[index_s[0:n2], :]).float()

    from scipy.stats import ortho_group
    S = np.float32(ortho_group.rvs(size=1, dim=X.shape[1], random_state=1))
    if scramble == 1:
        X = torch.matmul(X, torch.Tensor(S))

    return X, y

def generate_data_list(args):
    n1 = 1000000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb

    X_list, y_list = [], []
    for i, r in enumerate(args.r_list):
        X, y = data_generation(n1, args.num_list[i], ps, pvb, pv, args.r_list[i], args.scramble)
        X_list.append(X.to(args.device))
        y_list.append(y.to(args.device))
    return X_list, y_list


def generate_test_data_list(args):
    n1 = 1000000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb

    X_list, y_list = [], []
    for r in [-2.9, -2.7, -2.5, -2.3, -2.1, -1.9]:
        X, y = data_generation(n1, 1000, ps, pvb, pv, r, args.scramble)
        X_list.append(X.to(args.device))
        y_list.append(y.to(args.device))
    return X_list, y_list


class MLP(nn.Module):
    def __init__(self, m=1024):
        super().__init__()
        self.layer1 = nn.Linear(10, m)
        self.layer2 = nn.Linear(m, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def compute_num_params(model, verbose):
    num_params = 0
    for p in model.parameters():
        num_params += len(p.view(-1).detach().cpu().numpy())
    if verbose:
        print("Number of parameters is: %d" % num_params)
    return num_params


def compute_NTF(model, X, num_params, args):
    model.zero_grad()
    y = model(X).squeeze()
    ret = torch.zeros(len(y), num_params).to(args.device)
    for i, loss in (enumerate(y)):
        loss.backward(retain_graph=True)
        gradients = []
        for p in model.parameters():
            gradients.append(p.grad.view(-1))
        gradients = torch.cat(gradients, dim=-1) - torch.sum(ret, dim=0)
        assert len(gradients) == num_params
        ret[i, :] = gradients
    return ret.detach().cpu().numpy()


def main_Compute_NTF(args):
    X_list, _ = generate_data_list(args)
    X_list = torch.cat(X_list, dim=0)

    model = MLP().to(args.device)

    num_params = compute_num_params(model, False)

    NTF = compute_NTF(model, X_list, num_params, args)
    U, S, VT = randomized_svd(NTF, n_components=50, n_iter=10, random_state=42)
    print((np.mean(U[:1000, :], axis=0) - np.mean(U[1000:, :], axis=0))[0:21])
    return



class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class OLS:
    def __init__(self, X, y, args):
        self.model = LinearRegression(X.shape[1], 1)
        self.X = X
        self.y = y
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = args.device

    def to_cuda(self):
        self.model.cuda(self.device)
        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def train(self):
        self.model.weight_init()
        epochs = 3000

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(self.X)
            loss = self.loss(pred, self.y) \
                   + 1e-2 * torch.mean(torch.abs(self.model.linear.weight))
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return self.model.linear.weight.clone().cpu().detach(), self.model.linear.bias.clone().cpu().detach()

class Cluster:
    def __init__(self, feature, y, K, args):
        self.feature = feature.cpu()
        self.label = y.cpu()
        self.K = K
        self.args = args
        self.center = None
        self.bias = None
        self.domain = None

    # run weighted lasso for each cluster and get new coefs and biases
    def ols(self):
        for i in range(self.K):
            index = torch.where(self.domain == i)[0]
            tempx = (self.feature[index, :]).reshape(-1, self.feature.shape[1])
            tempy = (self.label[index, :]).reshape(-1, 1)
            clf = OLS(tempx, tempy, self.args)
            self.center[i, :], self.bias[i] = clf.train()

    def clustering(self, past_domains=None):
        # init
        self.center = torch.tensor(np.zeros((self.K, self.feature.shape[1]), dtype=np.float32))
        self.bias = torch.tensor(np.zeros(self.K, dtype=np.float32))

        # using last domains as the initialization
        if past_domains is None:
            self.domain = torch.tensor(np.random.randint(0, self.K, self.feature.shape[0]))
        else:
            self.domain = past_domains
        assert self.domain.shape[0] == self.feature.shape[0]

        # flags
        iter = 0
        end_flag = False
        delta_threshold = 0.1 * self.feature.shape[0]/self.K
        while not end_flag:
            iter += 1
            self.ols()
            ols_error = []

            for i in range(self.K):
                coef = self.center[i].reshape(-1, 1)
                error = torch.abs(torch.mm(self.feature, coef) + self.bias[i] - self.label)
                assert error.shape == (self.feature.shape[0], 1)
                ols_error.append(error)
            ols_error = torch.stack(ols_error, dim=0).reshape(self.K, self.feature.shape[0])

            new_domain = torch.argmin(ols_error, dim=0)
            assert new_domain.shape[0] == self.feature.shape[0]
            diff = self.domain.reshape(-1, 1) - new_domain.reshape(-1, 1)
            diff[diff != 0] = 1
            delta = torch.sum(diff)
            if iter % 10 == 9:
                print("Iter %d | Delta = %d" % (iter, delta))
            if delta <= delta_threshold:
                end_flag = True
            self.domain = new_domain


        return self.domain

def main_KernelHRM(args):
    print("Kernel HRM")

    class Linear_Model(nn.Module):
        def __init__(self, d=30):
            super().__init__()
            self.linear = nn.Linear(d, 1, bias=False)
            nn.init.xavier_uniform_(self.linear.weight, gain=0.1)

        def forward(self, f_w0, X):
            return f_w0 + self.linear(X)



    train_record = np.zeros(args.whole_epoch)
    test_record = np.zeros(args.whole_epoch)
    mean_stable_record = np.zeros(args.whole_epoch)
    std_stable_record = np.zeros(args.whole_epoch)
    # data
    X_list, y_list = generate_data_list(args)
    train_X, train_y = torch.cat([X_list[0], X_list[1]], dim=0), torch.cat([y_list[0], y_list[1]], dim=0)
    test_X, test_y = X_list[2], y_list[2]

    test_X_list, test_y_list = generate_test_data_list(args)

    model = MLP().to(args.device)
    init_params = torch.cat([p.view(-1) for p in model.parameters()], 0)
    criterion = torch.nn.MSELoss()
    NTF = compute_NTF(model, train_X, compute_num_params(model, False), args)
    test_NTF = compute_NTF(model, test_X, compute_num_params(model, False), args)
    U, S, VT = randomized_svd(NTF, n_components=args.k, n_iter=10, random_state=42)
    U, S, VT = torch.from_numpy(U).float().to(args.device), torch.from_numpy(S).float().to(
        args.device), torch.from_numpy(VT).float().to(args.device)
    U_train = torch.matmul(U, torch.diag(S))
    U_test = torch.from_numpy(test_NTF).float().to(args.device)
    U_test = torch.matmul(U_test, VT.permute(1, 0))
    train_feature = copy.deepcopy(U_train)
    U_train_sum = torch.sum(U_train.pow(2), dim=1)
    print(U_train_sum.shape)
    U_train_norm = torch.mean(torch.sqrt(U_train_sum))
    print("U_train norm is %.4f" % U_train_norm.data)


    tu_list = []
    for idx, tx in enumerate(test_X_list):
        tu = compute_NTF(model, tx, compute_num_params(model, False), args)
        tu = torch.from_numpy(tu).float().to(args.device)
        tu = torch.matmul(tu, VT.permute(1, 0))
        tu_list.append(tu)

    # whole iteration
    past_domains = None
    for whole_epoch in range(args.whole_epoch):
        print('--------------epoch %d---------------' % whole_epoch)
        # frontend
        cluster_model = Cluster(train_feature, train_y, args.cluster_num, args)
        cluster_results = cluster_model.clustering(past_domains)
        past_domains = cluster_results
        index0 = torch.where(cluster_results==0)[0]
        index1 = torch.where(cluster_results==1)[0]

        # calculate envs
        env_num_list = []
        for i in range(args.cluster_num):
            idx = torch.where(cluster_results[:1000, ] == i)[0]
            env_num_list.append(idx.shape[0])
        print('The first environment is split into : %s', pretty(env_num_list))

        env_num_list = []
        for i in range(args.cluster_num):
            idx = torch.where(cluster_results[1000:, ] == i)[0]
            env_num_list.append(idx.shape[0])
        print('The second environment is split into : %s', pretty(env_num_list))

        # backend
        flag = True
        theta_inv = None

        while flag:
            print("Step 1: Linear MIP")
            model_IRM = Linear_Model(d=U_train.shape[1]).to(args.device)
            model.eval()
            f_w0 = model(train_X).detach()
            opt_IRM = torch.optim.Adam(model_IRM.parameters(), lr=args.lr)

            for epoch in (range(1, args.epochs + 1)):
                model_IRM.train()

                yhat = model_IRM(f_w0[index0], U_train[index0,:])
                loss_1 = criterion(yhat, train_y[index0])
                grad_1 = grad(criterion(yhat, train_y[index0]), model_IRM.parameters(), create_graph=True)[0]

                yhat = model_IRM(f_w0[index1], U_train[index1,:])
                loss_2 = criterion(yhat, train_y[index1])
                grad_2 = grad(criterion(yhat, train_y[index1]), model_IRM.parameters(), create_graph=True)[0]

                penalty = (grad_1-grad_2).pow(2).mean()

                IRM_lam = args.IRM_lam if epoch > args.IRM_ann else 0.6
                loss = (loss_1 + loss_2) / 2 + IRM_lam * penalty

                opt_IRM.zero_grad()
                loss.backward()
                opt_IRM.step()

                model_IRM.eval()
                yhat = model_IRM(model(train_X), U_train)
                train_error = criterion(yhat, train_y)
                yhat = model_IRM(model(test_X), U_test)
                test_error = criterion(yhat, test_y)

                if epoch % 100 == 0:
                    print("Linear MIP epoch: %d, Train Error: %f, Test Error: %f" % (epoch, train_error, test_error))

            theta_inv = copy.deepcopy(model_IRM.linear.weight.data)
            flag = False
        train_record[whole_epoch] = train_error.data
        test_record[whole_epoch] = test_error.data
        theta_inv = theta_inv/(torch.sqrt(torch.sum(theta_inv.pow(2))))
        inner_product = torch.matmul(U_train, theta_inv.reshape(-1,1))
        assert inner_product.shape[1]==1 and inner_product.shape[0]==U_train.shape[0]
        train_feature = U_train - torch.matmul(inner_product, theta_inv.reshape(1,-1))

        print(train_feature.shape)

        # testing stage
        stable_test_error_list = []
        for idx, tu in enumerate(tu_list):
            model_IRM.eval()
            yhat = model_IRM(model(test_X_list[idx]), tu)
            s_error = criterion(yhat, test_y_list[idx])
            stable_test_error_list.append(s_error.data)
        stable_test_error_list = np.array(stable_test_error_list)
        mean_stable_error = np.mean(stable_test_error_list)
        std_stable_error = np.std(stable_test_error_list)
        mean_stable_record[whole_epoch] = mean_stable_error
        std_stable_record[whole_epoch] = std_stable_error
        print('Whole Epoch % d Mean %.4f Std %.4f' % (whole_epoch, np.mean(stable_test_error_list), np.std(stable_test_error_list)))


    return train_error.data, test_error.data, train_record, test_record, mean_stable_record, std_stable_record





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kernelized-HRM')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=60, help='k for SVD')
    parser.add_argument('--IRM_lam', type=float, default=6e1, help='IRM lambda')  
    parser.add_argument('--IRM_ann', type=int, default=500, help='IRM annealing') 
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--r_list', type=float, nargs='+', default=[0.8, 0.9, 0.1])
    parser.add_argument('--num_list', type=int, nargs='+', default=[1000, 1000, 1000])
    parser.add_argument('--method', type=str, default='KIRM')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--whole_epoch', type=int, default=5)
    parser.add_argument('--cluster_num', type=int, default=2)
    parser.add_argument('--scramble', type=int, default=0)
    args = parser.parse_args()

    args.device = torch.device("cuda:" + args.device if torch.cuda.is_available() and int(args.device)>0 else "cpu")

    setup_seed(args.seed)
    
    train_acc_list = []
    test_acc_list = []
    train_all = []
    test_all = []
    mean_all = []
    std_all = []
    for seed in range(9):
        print("-----------------seed %d ----------------" % seed)
        setup_seed(seed)
        result = main_KernelHRM(args)
        train_acc_list.append(result[0])
        test_acc_list.append(result[1])
        train_all.append(result[2])
        test_all.append(result[3])
        mean_all.append(result[4])
        std_all.append(result[5])
    train_acc_list = np.vstack(train_acc_list)
    test_acc_list = np.vstack(test_acc_list)

    print(train_acc_list)
    print(test_acc_list)

    print('===========mean=============')
    print(mean_all)
    print(std_all)


    print("MIP Train Mean %.4f  std %.4f" % (np.mean(train_acc_list), np.std(train_acc_list)))
    print("MIP Test Mean %.4f  std %.4f" % (np.mean(test_acc_list), np.std(test_acc_list)))

    print(train_all)
    print(test_all)

    print(np.mean(np.array(train_all), axis=0))
    print(np.std(np.array(train_all), axis=0))
    print(np.mean(np.array(test_all), axis=0))
    print(np.std(np.array(test_all), axis=0))

    print(np.mean(np.array(mean_all), axis=0))
    print(np.std(np.array(mean_all), axis=0))
    print(np.mean(np.array(std_all), axis=0))
    print(np.std(np.array(std_all), axis=0))







