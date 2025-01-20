import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from metrics import *
class Adversary():
    """
    Represents an adversary in a robust optimization framework, responsible for generating scenarios
    and optimizing risk measures.

    Attributes:
    rm_params : object
        Parameters related to risk measures.
    
    lm_params : object
        Parameters related to Lagrange multipliers.
    
    wass_params : object
        Parameters related to Wasserstein distance calculations.
    
    train_params : object
        Parameters related to training configurations.
    
    device : torch.device
        Device to perform calculations on (CPU or GPU).
    
    inner_net : nn.Module
        Neural network used to distort X_phi.
    
    net_name : str
        Name of the inner network.
    
    objective : str
        Optimization objective, either 'maximize' or 'minimize'.
    
    reset_lm : bool
        Flag to reset Lagrange multipliers after training.
    
    reset_net : bool
        Flag to reset the inner network after training.
    
    update_count : int
        Count of training iterations.
    
    loss_history : list
        History of loss values during training.
    
    rm_theta_history : list
        History of risk measures for X_theta.
    
    wass_dist_history : list
        History of Wasserstein distances.
    
    lam_history : list
        History of Lagrange multipliers.
    """

    def __init__(self, rm_params, lm_params, wass_params, train_params, device, inner_net, net_name, objective, reset_lm, reset_net):
        """
        Initializes the Adversary class with the given parameters.

        Parameters:
        rm_params : object
            Risk measure parameters.
        
        lm_params : object
            Lagrange multiplier parameters.
        
        wass_params : object
            Wasserstein distance parameters.
        
        train_params : object
            Training parameters.
        
        device : torch.device
            Device for calculations (CPU or GPU).
        
        inner_net : nn.Module
            Inner network responsible for distorting X_phi.
        
        net_name : str
            Name of the inner network.
        
        objective : str
            Objective for optimization ('maximize' or 'minimize').
        
        reset_lm : bool
            Flag to reset Lagrange multipliers after training.
        
        reset_net : bool
            Flag to reset the inner network after training.
        """
        self.rm_params = rm_params
        self.lm_params = lm_params
        self.wass_params = wass_params
        self.train_params = train_params
        self.device = device
        # Save the initial state of the inner network
        self.inner_net = inner_net
        torch.save(self.inner_net.state_dict(), "initial_inner")
        self.net_name = net_name
        _, _, _, _, inner_lr, _, _, _ = self.train_params.GetParams()
        self.optimizer = optim.Adam(self.inner_net.parameters(), lr=inner_lr)
        self.objective = objective
        self.reset_lm = reset_lm
        self.reset_net = reset_net
        self.update_count = 0
        self.InitHistory()

    def sim_theta(self, X_phi_T, market_model):
        """
        Generates terminal X_theta scenarios based on X_phi_T and the market model.

        Parameters:
        X_phi_T : torch.Tensor
            Terminal scenarios for X_phi.
        
        market_model : object
            The market model used for generating scenarios.

        Raises:
        Exception
            Must be overridden in subclass.
        """
        raise Exception("Must be overridden in subclass.")

    def train(self, X_phi_T, market_model):
        """
        Trains the inner network using the provided terminal scenarios and market model.

        Parameters:
        X_phi_T : torch.Tensor
            Terminal scenarios for X_phi.
        
        market_model : object
            The market model used for generating scenarios.

        Returns:
        nn.Module
            The trained inner network.
        """
        if self.reset_lm:
            self.lm_params.ResetParams()

        if self.reset_net:
            self.inner_net.load_state_dict(torch.load("initial_inner"))
            self.InitHistory()
            self.update_count = 0

        inner_epochs, _, plot_freq_inner, _, _, _, _, _ = self.train_params.GetParams()
        for i in range(inner_epochs):
            self.optimizer.zero_grad()
            X_theta_T = self.sim_theta(X_phi_T, market_model).reshape(-1, 1)
            loss, _, rm_theta, wass_dist = GetMetrics(X_phi_T, X_theta_T, self.rm_params, self.lm_params, self.wass_params, rm_objective=self.objective, problem_type="inner", device=self.device)
            loss.backward()
            self.optimizer.step()
            self.UpdateHistory(loss.item(), rm_theta, wass_dist)
            self.update_count += 1

            lam, mu, update_freq = self.lm_params.GetParams()
            if self.update_count % update_freq == 0:
                self.UpdateLM()

            if (i + 1) % plot_freq_inner == 0:
                print("Inner Epoch: {}".format(i))
                self.PlotCustom(X_phi_T, X_theta_T)

        return self.inner_net

    def UpdateLM(self):
        """
        Updates the Lagrange multipliers based on recent constraint errors.
        """
        _, wass_limit = self.wass_params.GetParams()
        lookback = min(len(self.wass_dist_history), 5)
        lam, mu = self.lm_params.UpdateParams(np.mean(np.array(self.wass_dist_history[-lookback:]) - wass_limit))
        self.lam_history.append(lam)

    def InitHistory(self):
        """
        Initializes history for tracking metrics during training.
        """
        self.loss_history = []
        self.rm_theta_history = []
        self.wass_dist_history = []
        self.lam_history = [self.lm_params.params["lam"]]

    def UpdateHistory(self, loss, rm_theta, wass_dist):
        """
        Updates the history of metrics.

        Parameters:
        loss : float
            The loss value to add to history.
        
        rm_theta : float
            The risk measure of X_theta to add to history.
        
        wass_dist : float
            The Wasserstein distance to add to history.
        """
        self.loss_history.append(loss)
        self.rm_theta_history.append(rm_theta)
        self.wass_dist_history.append(wass_dist)

    def PrintMetrics(self):
        """
        Prints the most recent metrics from training history.
        """
        lookback = min(len(self.wass_dist_history), 5)
        print("Wass Dist History: ", self.wass_dist_history[-lookback:])
        print("Risk Measure X_theta History: ", self.rm_theta_history[-lookback:])
        print("Loss History: ", self.loss_history[-lookback:])
        lam, mu, _ = self.lm_params.GetParams()
        print("Augmented Lagrangian lambda: {} mu: {}".format(lam, mu))

    def PlotCustom(self, X, Y):
        """
        Placeholder for custom plotting functionality. Can be overridden in subclasses.

        Parameters:
        X : torch.Tensor
            Input tensor for plotting.
        
        Y : torch.Tensor
            Input tensor for plotting.
        """
        pass

    def PlotHistory(self):
        """
        Plots the history of loss, risk measures, Wasserstein distances, and Lagrange multipliers.
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.rm_theta_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel(r'$R[X_{\theta}]$', fontsize=20)

        plt.subplot(1, 3, 2)
        plt.plot(self.wass_dist_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Wass Distance", fontsize=20)

        plt.subplot(1, 3, 3)
        plt.plot(self.lam_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel(r'$\lambda$', fontsize=20)
        
        plt.tight_layout()
        plt.show()

    def PlotDistortion(self, X, Y):
        """
        Plots the distribution of X_phi and X_theta, as well as their relationship.

        Parameters:
        X : torch.Tensor
            Terminal scenarios for X_phi.
        
        Y : torch.Tensor
            Terminal scenarios for X_theta.
        """
        plt.figure(figsize=(15, 5))
        a, b, p, rm_type = self.rm_params.GetParams()
        X = X.cpu().detach().numpy().squeeze()
        Y = Y.cpu().detach().numpy().squeeze()

        ax1 = plt.subplot(1, 3, 1)
        # 绘制 X 的直方图和 KDE
        sns.histplot(X, kde=True, label=r'$X^{\phi}$', stat="density", bins=30, color='deepskyblue', alpha=0.5,edgecolor='none')
        sns.kdeplot(X, color='blue')

        # 绘制 Y 的直方图和 KDE
        sns.histplot(Y, kde=True, label=r'$X^{\theta}$', stat="density", bins=30, color='#FF4500', alpha=0.5,edgecolor='none')
        sns.kdeplot(Y, color='#FF4500')
        # sns.distplot(X, hist=True, kde=True, label=r'$X^{\phi}$')
        # sns.distplot(Y, hist=True, kde=True, label=r'$X^{\theta}$')
        plt.axvline(np.mean(X[X <= np.quantile(X, a)]), color='k', ls='--', alpha=0.35)
        plt.axvline(np.mean(Y[Y <= np.quantile(Y, a)]), color='r', ls='--', alpha=0.35)

        if rm_type == 'alpha-beta':
            plt.axvline(np.mean(X[X > np.quantile(X, b)]), color='k', ls='--', alpha=0.35)
            plt.axvline(np.mean(Y[Y > np.quantile(Y, b)]), color='r', ls='--', alpha=0.35)
        elif rm_type == 'custom':
            plt.axvline(np.mean(X), color='k', ls='--', alpha=0.35)
            plt.axvline(np.mean(Y), color='r', ls='--', alpha=0.35)

        plt.xlabel(r'$X_{T}$', fontsize=20)
        plt.legend(loc="best", fontsize=20)

        ax2 = plt.subplot(1, 3, 2)
        base = [min(X), max(X)]
        plt.plot(base, base, '--k')
        plt.scatter(X, Y, marker='o', color='r', s=0.5, alpha=0.25)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.xlabel(r'$X_{T}^{\phi}$', fontsize=20)
        plt.ylabel(r'$X_{T}^{\theta}$', fontsize=20)

        ax3 = plt.subplot(1, 3, 3)
        ecdf_Y = ECDF(Y)
        U_Y = ecdf_Y(Y)
        ecdf_X = ECDF(X)
        U_X = ecdf_X(X)
        plt.scatter(U_Y, Y, label=r"$X^{\theta}$ ", s=2, alpha=1)
        plt.plot(np.sort(U_X), np.sort(X), '--k', label=r"$X^{\phi}$")
        plt.xlabel("Quantiles", fontsize=20)
        plt.ylabel(r'$X_{T}$', fontsize=20)
        plt.legend(loc="best", fontsize=20)
        
        plt.tight_layout()
        plt.show()


class Agent():
    """
    Represents an agent that controls the outer problem in a robust optimization framework.
    The agent aims to find the best robust strategy given a market model and an adversary.

    Attributes:
    market_model : object
        The market model used for simulations.
    
    sim_params : object
        Parameters related to simulation settings.
    
    train_params : object
        Parameters related to training configurations.
    
    device : torch.device
        Device for calculations (CPU or GPU).
    
    requires_update : bool
        Indicates if the agent needs to be trained.
    
    outer_net : nn.Module
        Neural network seeking a robust strategy over the market model.
    
    net_name : str
        Name of the outer network.
    
    X_theta_history : np.ndarray
        History of terminal X_theta scenarios.
    
    other_loss : torch.Tensor
        Other sources of loss not pertaining to risk measures.
    """

    def __init__(self, market_model, sim_params, train_params, device, outer_net, net_name, requires_update):
        """
        Initializes the Agent class with the given parameters.

        Parameters:
        market_model : object
            The market model used for simulations.
        
        sim_params : object
            Simulation parameters.
        
        train_params : object
            Training parameters.
        
        device : torch.device
            Device for calculations (CPU or GPU).
        
        outer_net : nn.Module
            Neural network seeking a robust strategy.
        
        net_name : str
            Name of the outer network.
        
        requires_update : bool
            Indicates if the agent needs to be trained.
        """
        self.market_model = market_model
        self.sim_params = sim_params
        self.train_params = train_params
        self.device = device
        self.requires_update = requires_update
        self.outer_net = outer_net
        self.net_name = net_name
        
        if self.requires_update:
            _, _, _, _, _, outer_lr, _, _ = self.train_params.GetParams()
            self.optimizer = optim.Adam(self.outer_net.parameters(), lr=outer_lr)

        _, _, Nsims, _, _, _, _ = self.sim_params.GetParams()
        self.X_theta_history = np.empty((Nsims, 0))
        self.other_loss = torch.zeros(1).to(device)

    def sim_phi(self):
        """
        Simulates terminal X_phi scenarios based on the outer network and market model.
        Needs to be overridden in subclasses.

        Raises:
        Exception
            Must be overridden in subclass.
        """
        raise Exception("Must be overridden in subclass.")

    def step_theta(self, X_phi_T, adversary):
        """
        Given X_phi_T and the adversary, finds X_theta_T and backpropagates the outer loss if necessary.

        Parameters:
        X_phi_T : torch.Tensor
            Terminal scenarios for X_phi.
        
        adversary : Adversary
            The adversary used to distort X_phi.

        Returns:
        torch.Tensor
            The terminal scenarios for X_theta.
        """
        if self.requires_update:
            self.optimizer.zero_grad()
            # Freeze the inner network parameters
            for param in adversary.inner_net.parameters():
                param.requires_grad = False
            
            # Use the trained inner network to distort X_phi
            X_theta_T = adversary.inner_net(X_phi_T).reshape(-1, 1)

            # Calculate loss using risk measures
            rm_loss, rm_phi, rm_theta, _ = GetMetrics(X_phi_T, X_theta_T, adversary.rm_params, adversary.lm_params, adversary.wass_params,
                                                      rm_objective="minimize", problem_type="outer", device=self.device)
            self.loss = rm_loss + self.other_loss
            self.loss.backward()
            self.optimizer.step()

            # Unfreeze the inner network parameters
            for param in adversary.inner_net.parameters():
                param.requires_grad = True
        else:
            # Find X_theta_T without backpropagation
            X_theta_T = adversary.sim_theta(X_phi_T, self.market_model).reshape(-1, 1)
            rm_theta = GetRiskMeasure(X_theta_T, adversary.rm_params).item()
            rm_phi = GetRiskMeasure(X_phi_T, adversary.rm_params).item()

        self.UpdateHistory(rm_theta, rm_phi)
        self.X_theta_history = np.concatenate((self.X_theta_history, X_theta_T.cpu().detach().numpy()), axis=1)
        return X_theta_T

    def train(self, adversary):
        """
        Trains the agent by simulating the market model and updating the outer network.

        Parameters:
        adversary : Adversary
            The adversary used for training.
        
        Returns:
        nn.Module
            The trained outer network.
        """
        self.InitHistory()
        _, outer_epochs, _, plot_freq_outer, _, _, freeze_market_iter, freeze_inner_iter = self.train_params.GetParams()
        for i in range(outer_epochs):
            self.market_model.Sim(self.sim_params)
            # Freeze market randomness for 'freeze_market_iter' iterations
            for j in range(freeze_market_iter):
                X_phi_T = self.sim_phi().reshape(-1, 1)
                curr_iter = i * freeze_market_iter + j
                
                # Freeze inner network for 'freeze_inner_iter' iterations
                if curr_iter % freeze_inner_iter == 0:
                    adversary.train(X_phi_T.detach(), self.market_model)
                    torch.save(adversary.inner_net.state_dict(), adversary.net_name + "_Epoch_{}_Iter_{}".format(i, j))
                
                # Find X_theta_T with trained adversary
                X_theta_T = self.step_theta(X_phi_T, adversary)
                if curr_iter % plot_freq_outer == 0:
                    print("Outer Epoch: {} Outer Iter: {}".format(i, j))
                    self.PlotCustom(X_phi_T, X_theta_T, adversary)
                    if self.requires_update:
                        torch.save(self.outer_net.state_dict(), self.net_name + "_Epoch_{}_Iter_{}".format(i, j))
        
        return self.outer_net

    def InitHistory(self):
        """
        Initializes history for tracking metrics during training.
        """
        self.other_loss_history = []
        self.rm_theta_history = []
        self.rm_phi_history = []

    def UpdateHistory(self, rm_theta, rm_phi):
        """
        Updates the history of risk measures.

        Parameters:
        rm_theta : float
            The risk measure of X_theta.
        
        rm_phi : float
            The risk measure of X_phi.
        """
        self.other_loss_history.append(self.other_loss.item())
        self.rm_theta_history.append(rm_theta)
        self.rm_phi_history.append(rm_phi)

    def PlotCustom(self, X, Y, adversary):
        """
        Placeholder for custom plotting functionality. Can be overridden in subclasses.

        Parameters:
        X : torch.Tensor
            Input tensor for plotting.
        
        Y : torch.Tensor
            Input tensor for plotting.
        
        adversary : Adversary
            The adversary involved in the plotting.
        """
        pass

    def PlotHistory(self):
        """
        Plots the history of losses and risk measures.
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.other_loss_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Other Loss", fontsize=20)

        plt.subplot(1, 3, 2)
        plt.plot(self.rm_theta_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel(r'$R[X_{\theta}]$', fontsize=20)

        plt.subplot(1, 3, 3)
        plt.plot(self.rm_phi_history)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel(r'$R[X_{\phi}]$', fontsize=20)
        
        plt.tight_layout()
        plt.show()
