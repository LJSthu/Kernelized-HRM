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

np.set_printoptions(precision=4)

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_data(num_data=10000, d=5, bias=0.5, scramble=0, sigma_s=3.0, sigma_v=0.3):
    from scipy.stats import ortho_group
    S = np.float32(ortho_group.rvs(size=1, dim=2*d, random_state=1))
    y = np.random.choice([1, -1], size=(num_data, 1))
    X = np.random.randn(num_data, d * 2)
    X[:, :d] *= sigma_s
    X[:, d:] *= sigma_v
    flip = np.random.choice([1, -1], size=(num_data, 1), p=[bias, 1. - bias]) * y
    X[:, :d] += y
    X[:, d:] += flip
    if scramble == 1:
        X = np.matmul(X, S)
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    return X, y


def generate_data_list(args):
    X_list, y_list = [], []
    for i, r in enumerate(args.r_list):
        X, y = generate_data(num_data=args.num_list[i], bias=r, scramble=args.scramble)
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
        # self.to_cuda()
        self.model.weight_init()
        epochs = 3000

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(self.X)
            loss = self.loss(pred, self.y) \
                   + 1e-2 * torch.mean(torch.abs(self.model.linear.weight))
            loss.backward(retain_graph=True)
            self.optimizer.step()
        #     if epoch % 100 == 0:
        #         print("Epoch %d | Loss = %.4f" % (epoch, loss))
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
            # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.linear.weight, gain=0.1)

        def forward(self, f_w0, X):
            return f_w0 + self.linear(X)



    train_record = []
    test_record = []
    # data
    X_list, y_list = generate_data_list(args)
    train_X, train_y = torch.cat([X_list[0], X_list[1]], dim=0), torch.cat([y_list[0], y_list[1]], dim=0)
    test_X, test_y = X_list[2], y_list[2]

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

    # whole iteration
    past_domains = None
    for epoch in range(args.whole_epoch):
        print('--------------epoch %d---------------' % epoch)
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
        correct1 = 0.0
        correct2 = 0.0
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
                scale = torch.tensor(1.).to(args.device).requires_grad_()

                yhat = model_IRM(f_w0[index0], U_train[index0,:])
                loss_1 = criterion(yhat, train_y[index0])
                grad_1 = grad(criterion(yhat * scale, train_y[index0]), [scale], create_graph=True)[0]

                yhat = model_IRM(f_w0[index1], U_train[index1,:])
                loss_2 = criterion(yhat, train_y[index1])
                grad_2 = grad(criterion(yhat * scale, train_y[index1]), [scale], create_graph=True)[0]

                penalty = (grad_1-grad_2).pow(2).mean()

                IRM_lam = args.IRM_lam if epoch > args.IRM_ann else 1.0
                loss = (loss_1 + loss_2) / 2 + IRM_lam * penalty

                opt_IRM.zero_grad()
                loss.backward()
                opt_IRM.step()

                pred_train = 2 * ((model_IRM(f_w0, U_train) > 0).float() - 0.5)
                correct1 = float(pred_train[index0].eq(train_y[index0]).sum().item())/len(index0)
                correct2 = float(pred_train[index1].eq(train_y[index1]).sum().item())/len(index1)


                correct = float(pred_train.eq(train_y).sum().item())
                total = pred_train.size(0)
                train_acc = correct / total

                model_IRM.eval()
                yhat = model_IRM(model(test_X), U_test)
                pred_test = 2 * ((yhat > 0).float() - 0.5)
                correct = float(pred_test.eq(test_y).sum().item())
                total = pred_test.size(0)
                test_acc = correct / total

                if epoch % args.epochs == 0:
                    print("Linear MIP epoch: %d, Train Acc: %f, Test Acc: %f" % (epoch, train_acc, test_acc))
                    print("Env 1 %.4f  Env 2 %.4f" % (correct1, correct2))
                
            theta_inv = copy.deepcopy(model_IRM.linear.weight.data)
            flag = False
            
        train_record.append(train_acc)
        test_record.append(test_acc)
        theta_inv = theta_inv/(torch.sqrt(torch.sum(theta_inv.pow(2))))
        print(theta_inv)
        print(torch.sum(theta_inv.pow(2)))
        inner_product = torch.matmul(U_train, theta_inv.reshape(-1,1))
        assert inner_product.shape[1]==1 and inner_product.shape[0]==U_train.shape[0]
        train_feature = U_train - torch.matmul(inner_product, theta_inv.reshape(1,-1))

        print(train_feature.shape)

    return train_acc, test_acc, train_record, test_record





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kernelized-HRM')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=60, help='k for SVD')
    parser.add_argument('--IRM_lam', type=float, default=6e1, help='IRM lambda')  # 3-5e1 for IGD # 1e3 for IRM
    parser.add_argument('--IRM_ann', type=int, default=500, help='IRM annealing')  # 200-400 for IGD, 400 for IRM
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--r_list', type=float, nargs='+', default=[0.8, 0.9, 0.1])
    parser.add_argument('--num_list', type=int, nargs='+', default=[1000, 1000, 1000])
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
    for seed in range(10):
        print("-----------------seed %d ----------------" % seed)
        setup_seed(seed)
        result = main_KernelHRM(args)
        train_acc_list.append(result[0])
        test_acc_list.append(result[1])
        train_all.append(result[2])
        test_all.append(result[3])
    train_acc_list = np.array(train_acc_list)
    test_acc_list = np.array(test_acc_list)
    print("MIP Train Mean %.4f  std %.4f" % (np.mean(train_acc_list), np.std(train_acc_list)))
    print("MIP Test Mean %.4f  std %.4f" % (np.mean(test_acc_list), np.std(test_acc_list)))
    print(train_all)
    print(test_all)
    print(np.mean(np.array(train_all), axis=0))
    print(np.std(np.array(train_all), axis=0))
    print(np.mean(np.array(test_all), axis=0))
    print(np.std(np.array(test_all), axis=0))













