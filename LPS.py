import math
import time
import torch
import random
import os
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs
from torch.distributions import Normal

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser(description='hyper parameters')
# ==============equation=========
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=0.4)
parser.add_argument('--L0', type=float, default=0)
parser.add_argument('--T0', type=float, default=0)
parser.add_argument('--L1', type=float, default=20)
parser.add_argument('--T1', type=float, default=1)
# ==============model============
parser.add_argument('--input_dims', type=int, default=2, help='input_dims')
parser.add_argument('--middle_dims', type=int, default=50, help='width')
parser.add_argument('--output_dims', type=int, default=1, help='output_dims')
parser.add_argument('--depth', type=int, default=8, help='depth')
# ==============data============
parser.add_argument('--nums_boundary', type=int, default=500, help='sampling boundary points')
parser.add_argument('--nums_internal', type=int, default=50, help='sampling internal points')
parser.add_argument('--nums_test_s', type=int, default=10, help='sampling testing space points')
parser.add_argument('--nums_test_t', type=int, default=20, help='sampling testing time points')
# ==============train===========
parser.add_argument('--epochs', type=int, default=200000, help='Epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step_size', type=int, default=1000, help='step_size')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--write_step', type=int, default=100, help='write_step')


class Model(nn.Module):
    def __init__(self,
                 in_dims: int,
                 middle_dims: int,
                 out_dims: int,
                 depth: int):
        super(Model, self).__init__()

        self.linearIn = nn.Linear(in_dims, middle_dims)

        self.linear = nn.ModuleList()
        for _ in range(depth):
            linear = nn.Linear(middle_dims, middle_dims)
            self.linear.append(linear)

        self.linearOut = nn.Linear(middle_dims, out_dims)

        self.act = nn.Tanh()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.act(self.linearIn(x))

        for layer in self.linear:
            out = self.act(layer(out))

        out = self.linearOut(out)

        return out


class DataGen(nn.Module):
    def __init__(self,
                 r: float = 0.1,
                 sigma: float = 0.5,
                 nums_boundary: int = 1000,
                 nums_internal: int = 2000,
                 L0: int = 0,
                 L1: int = 20,
                 T0: int = 0,
                 T1: int = 1,
                 # length_s: int = 20,
                 # length_t: int = 1,
                 grid_num_s: int = 51,
                 grid_num_t: int = 50,
                 device: torch.device = 'cuda:0'):
        super(DataGen, self).__init__()
        self.r = r
        self.sigma = sigma
        self.nums_boundary = nums_boundary
        self.nums_internal = nums_internal
        # self.length_s = length_s
        # self.length_t = length_t
        self.L0 = L0
        self.L1 = L1
        self.T0 = T0
        self.T1 = T1
        self.grid_num_s = grid_num_s
        self.grid_num_t = grid_num_t
        self.device = device

    def gen_train_data(self, ):
        si = 0.0000001

        bc1_coord = np.array([[self.L0, self.T0], [self.L1, self.T0 ]])  # initial time
        bc2_coord = np.array([[self.L1, self.T0], [self.L1, self.T1]])  # bc
        bc3_coord = np.array([[self.L1, self.T1], [self.L0, self.T1]])  # bc
        bc4_coord = np.array([[self.L0, self.T1], [self.L0, self.T0]])  # bc

        brdy_1 = self.sample(bc1_coord, self.nums_boundary)
        brdy_2 = self.sample(bc2_coord, self.nums_boundary)
        brdy_3 = self.sample(bc3_coord, self.nums_boundary)
        brdy_4 = self.sample(bc4_coord, self.nums_boundary)

        x_ic = torch.from_numpy(brdy_3).float().requires_grad_(True).to(self.device)  # init condition
        x_bc0 = torch.from_numpy(brdy_4).float().requires_grad_(True).to(self.device)
        x_bc = np.concatenate([brdy_1, brdy_2], axis=0)
        x_bc = torch.from_numpy(x_bc).float().requires_grad_(True).to(self.device)  # bc4
        x_bc_exact = self.exact_solution(x_bc).to(self.device)

        lb = np.array([self.L0 + si, self.T0 + si])
        ub = np.array([self.L1, self.T1])
        data_train = lb + (ub - lb) * lhs(2, self.nums_internal)
        data_train = torch.from_numpy(data_train).float().requires_grad_(True).to(self.device)
        data_train_exact = self.exact_solution(data_train).to(self.device)

        return data_train, x_ic, x_bc0, x_bc, data_train_exact, x_bc_exact

    def gen_test_data(self, ):

        s = np.linspace(self.L0, self.L1, self.grid_num_s)[1:-1]
        t = np.linspace(self.T0, self.T1, self.grid_num_t)[1:-1]
        S, T = np.meshgrid(s, t)

        data_test = np.concatenate([S.flatten()[:, None], T.flatten()[:, None]], axis=1)
        data_test = torch.from_numpy(data_test).float()
        data_test = data_test.requires_grad_(True).to(self.device)

        data_test_exact = self.exact_solution(data_test)

        return data_test, data_test_exact

    def exact_solution(self, x):
        K = 10
        T = self.T1

        S = x[:, 0:1]
        t = x[:, 1:2]

        dd1 = (torch.log(S / K) + (self.r + 0.5 * (self.sigma ** 2)) * (T - t)) / (self.sigma * torch.sqrt(T - t))
        dd2 = dd1 - self.sigma * torch.sqrt(T - t)
        d1 = torch.squeeze(dd1, 1)
        d2 = torch.squeeze(dd2, 1)
        normal = Normal(loc=0, scale=1)  # mean = loc, stddev = scale
        N_d1 = normal.cdf(value=d1).unsqueeze(1)  # Cumulative distribution function: cdf(d): P{X<d}, X~N(loc, scale)
        N_d2 = normal.cdf(value=d2).unsqueeze(1)
        output = S * N_d1 - K * torch.exp(-self.r * (T - t)) * N_d2

        return output

    def operator(self, V, X, kind='inter'):
        K = 10

        S = X[:, 0:1]
        V_t = self.get_gradients(V, X)[:, 1:2]
        V_s = self.get_gradients(V, X)[:, 0:1]
        V_ss = self.get_gradients(V_s, X)[:, 0:1]
        if kind == 'inter':
            output = V_t + 1/2 * (self.sigma**2) * (S ** 2) * V_ss - self.r * V + self.r * S * V_s
        elif kind == 'icT':
            output = V - torch.nn.functional.relu(S-K)
        elif kind == 'icS0':
            output = V - 0
        else:
            output = 0
            print("Nothing can do!")

        return output.to(self.device)

    def conservation(self, u, data, a=1, b=1):
        u = u.squeeze(1)
        x = data[:, 0]
        t = data[:, 1]
        lt = 1 * t
        lt_t = self.get_gradients(lt, data)[:, 1]
        gt = t ** t
        A = self.sigma
        B = self.r
        u_x = self.get_gradients(u, data)[:, 0]
        u_t = self.get_gradients(u, data)[:, 1]
        T1 = -u_x * lt + a / x + (2 * b * u / (A ** 2 * x)) * torch.exp(-B * t)
        T2 = u_t * lt + u * lt_t + gt - b * u * torch.exp(-B * t) + b * (
                    u_x + (2 * B * u) / (A ** 2 * x)) * x * torch.exp(-B * t)

        output_jxl0 = self.get_gradients(T1, data)[:, 1] + self.get_gradients(T2, data)[:, 0]

        return output_jxl0

    def NIPS(self, u, data):
        u = u.squeeze(1)
        x = data[:, 0]
        t = data[:, 1]
        u_x = self.get_gradients(u, data)[:, 0]
        u_t = self.get_gradients(u, data)[:, 1]
        u_tx = self.get_gradients(u_t, data)[:, 0]
        u_xx = self.get_gradients(u_x, data)[:, 0]
        u_xxx = self.get_gradients(u_xx, data)[:, 0]
        nips_out = x * u_tx + self.r * (x ** 2) * u_xx - self.r * x * u_x + 0.5 * (self.sigma ** 2) * (x ** 3) * u_xxx

        return nips_out

    @staticmethod
    def get_gradients(f, x):
        grad = torch.autograd.grad(
            f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True,
            allow_unused=True)[0]
        return grad

    @staticmethod
    def sample(coord, N):
        dim = coord.shape[1]
        x = coord[0:1, :] + (coord[1:2, :] - coord[0:1, :]) * np.random.rand(N, dim)
        return x

    @staticmethod
    def error_fun(output, target):
        error = output - target
        error = math.sqrt(torch.mean(error * error))
        ref = math.sqrt(torch.mean(target * target))
        return error / (ref + 10 ** (-8))


def main(lambda_r,lambda_bc,lambda_ic,lambda_con,r,sigma):
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model
    model = Model(
        in_dims=args.input_dims,
        middle_dims=args.middle_dims,
        out_dims=args.output_dims,
        depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    # init data
    data_gen = DataGen(
        r=r,
        sigma=sigma,
        nums_boundary=args.nums_boundary,
        nums_internal=args.nums_internal,
        L0=args.L0,
        L1=args.L1,
        T0=args.T0,
        T1=args.T1,
        grid_num_s=args.nums_test_s,
        grid_num_t=args.nums_test_t,
        device=device
    )

    data_test, data_test_exact = data_gen.gen_test_data()
    x_r, x_ic, x_bc0, x_bc, x_r_exact, x_bc_exact = data_gen.gen_train_data()

    #
    Test_relative_error = []
    Conservation_error_test = []
    Conservation_error_exact = []

    folder='./LPS_r{rr}_sigma{sigma}_l1{l1}_l2{l2}_l3{l3}_l4{l4}'.format(
        rr=r,
        sigma=sigma,
        l1=lambda_r,
        l2=lambda_bc,
        l3=lambda_ic,
        l4=lambda_con)
    os.makedirs(folder, exist_ok=True)
    #
    start_time = time.time()
    for step in range(args.epochs):
        y_r = model(x_r)
        y_ic = model(x_ic)
        y_bc0 = model(x_bc0)
        y_bc = model(x_bc)

        loss_r = torch.mean(data_gen.operator(y_r, x_r, kind='inter') ** 2)
        loss_ic = torch.mean(data_gen.operator(y_ic, x_ic, kind='icT') ** 2)
        loss_bc0 = torch.mean(data_gen.operator(y_bc0, x_bc0, kind='icS0') ** 2)
        loss_bc = torch.mean((y_bc - x_bc_exact) ** 2)

        # # =================Law of Conservation===========================================
        loss_con0 = data_gen.conservation(y_r, x_r)
        loss_con_exact = data_gen.conservation(x_r_exact, x_r)
        loss_con = torch.mean(loss_con0 ** 2)
        loss_con_ver = torch.mean(loss_con_exact ** 2)

        loss_con0_lps = data_gen.NIPS(y_r, x_r)
        loss_con_exact_lps = data_gen.NIPS(x_r_exact, x_r)
        loss_con_lps = torch.mean(loss_con0_lps ** 2)
        loss_con_ver_lps = torch.mean(loss_con_exact_lps ** 2)

        losses = (lambda_r * loss_r + lambda_bc * (loss_bc + loss_bc0) +
                  lambda_ic * loss_ic + lambda_con * loss_con_lps)

        optimizer.zero_grad()
        model.zero_grad()
        losses.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # ==============================================================================================================
        if step % args.write_step == 0:
            elapsed = time.time() - start_time
            start_time = time.time()

            data_test_pred = model(data_test)
            loss_con0_test = data_gen.conservation(data_test_pred, data_test)
            loss_con_exact_test = data_gen.conservation(data_test_exact, data_test)
            loss_con_test = torch.mean(loss_con0_test ** 2)
            loss_con_ver_test = torch.mean(loss_con_exact_test ** 2)

            test_error = F.mse_loss(data_test_pred, data_test_exact)
            abs_test_error = torch.max(torch.abs(data_test_pred - data_test_exact))
            relative_test_error = data_gen.error_fun(data_test_pred, data_test_exact)

            print('Epoch: %d, Time: %.2f, Loss_con: %.3e, Loss_con_lps: %.3e, Loss: %.3e,Loss_bc: %.3e, Loss_ic: %.3e, Loss_r: %.3e, Test error: %.3e, Relative Test error: %.3e, abs Test error: %.3e' %
                (step, elapsed, loss_con, loss_con_lps, losses, loss_bc, loss_ic, loss_r, test_error, relative_test_error, abs_test_error))




            Test_relative_error.append(relative_test_error)
            Conservation_error_test.append(loss_con_test.cpu().detach().numpy())
            Conservation_error_exact.append(loss_con_ver_test.cpu().detach().numpy())

    u_num_test = model(data_test)
    U_numerical_test = u_num_test.cpu().detach().numpy()
    U_test = data_test_exact.cpu().detach().numpy()

    Test_relative_error = [t for t in Test_relative_error]
    np.savetxt(folder + "/Test_relative_error.csv", Test_relative_error)

    Conservation_error_test = [t for t in Conservation_error_test]
    np.savetxt(folder + "/Test_conservation_num.csv", Conservation_error_test)
    Conservation_error_exact = [t for t in Conservation_error_exact]
    np.savetxt(folder + "/Test_conservation_exact.csv", Conservation_error_exact)

    np.savetxt(folder + "/Numerical_test.txt", U_numerical_test)
    np.savetxt(folder + "/exact_test.txt", U_test)


if __name__ == "__main__":

            
    lambda_r = 0.001
    lambda_bc = 1
    lambda_ic = 0.1
    lambda_con = 1
    r = 0.1
    sigma = 0.4

    main(lambda_r,lambda_bc,lambda_ic,lambda_con,r,sigma)
