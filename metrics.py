import torch

def GetRiskMeasure(X, rm_params):
    """
    Calculates risk measures based on different types: "alpha-beta" or "mean-CVaR".

    Parameters:
    X : torch.Tensor
        Input tensor, typically representing asset return data with shape (Nsims, Nassets),
        where Nsims is the number of simulations and Nassets is the number of assets.
    
    rm_params : object
        An object containing parameters related to risk measures, usually including:
        - alpha: the alpha quantile for calculating the lower bound of the risk measure.
        - beta: the beta quantile for calculating the upper bound of the risk measure.
        - p: risk aversion parameter used for weighting.
        - rm_type: the type of risk measure, either "alpha-beta" or "mean-CVaR".

    Returns:
    RiskMeasure : float
        The calculated risk measure based on the specified type.
    """
    
    # Extract parameters from rm_params using GetParams()
    alpha, beta, p, rm_type = rm_params.GetParams()

    # Calculate the alpha-beta risk measure of batch X
    if rm_type == "alpha-beta":
        # Find the alpha and beta quantiles of X
        LQtl = torch.quantile(X, alpha)  # Lower quantile
        UQtl = torch.quantile(X, beta)    # Upper quantile
        # Normalizing factor
        eta = p * alpha + (1 - p) * (1 - beta)
        # Calculate risk measure by computing the weighted average of values between quantiles
        RiskMeasure = -(p * alpha * torch.mean(X[X <= LQtl]) + (1 - p) * (1 - beta) * torch.mean(X[X >= UQtl])) / eta

    # Calculate mean-CVaR risk measure of batch X
    elif rm_type == "mean-CVaR":
        # Mean-CVaR risk measure combines mean and CVaR, where CVaR computes tail losses below the alpha quantile.
        LQtl = torch.quantile(X, alpha)
        # Weighted combination of mean and CVaR with 10x more emphasis on the CVaR
        RiskMeasure = -1 / 11 * torch.mean(X) - 10 / 11 * torch.mean(X[X <= LQtl])

    else:
        print("Risk Measure Type Not Supported.")
  
    return RiskMeasure

def GetMetrics(X, Y, rm_params, lm_params, wass_params, rm_objective='maximize', problem_type='inner', device=torch.device('cpu')):
    """
    Computes various metrics including loss, risk measures of X and Y, and Wasserstein distance.

    Parameters:
    X : torch.Tensor
        Input tensor representing asset return data for the first variable.
    
    Y : torch.Tensor
        Input tensor representing asset return data for the second variable.
    
    rm_params : object
        An object containing parameters related to risk measures, including:
        - alpha: the alpha quantile for calculating lower risk measure bounds.
        - beta: the beta quantile for calculating upper risk measure bounds.
        - p: risk aversion parameter for weighting.
        - rm_type: type of risk measure, either "alpha-beta" or "mean-CVaR".

    lm_params : object
        An object containing parameters related to Lagrange multipliers for constraints.

    wass_params : object
        An object containing parameters for Wasserstein distance calculation, including:
        - wass_order: the order of the Wasserstein distance.
        - wass_limit: the limit for Wasserstein distance constraints.

    rm_objective : str, optional
        Objective for risk measure optimization, either 'maximize' or 'minimize'. Default is 'maximize'.

    problem_type : str, optional
        Type of optimization problem, either 'inner' or 'outer'. Default is 'inner'.

    device : torch.device, optional
        Device to perform calculations on (CPU or GPU). Default is CPU.

    Returns:
    tuple
        A tuple containing:
        - loss : float
            The computed loss value.
        - rm_phi : float
            The risk measure of X.
        - rm_theta : float
            The risk measure of Y.
        - wass_dist : float
            The Wasserstein distance between X and Y.
    """
    Nsims = X.shape[0]
    # Sort both X and Y so X_sorted and Y_sorted are comonotonic
    X_sorted, _ = torch.sort(X, dim=0)
    Y_sorted, _ = torch.sort(Y, dim=0)
    X_sorted_nograd = X_sorted.detach()
    Y_sorted_nograd = Y_sorted.detach()
    
    # Calculate the gradient of the distribution function of Y
    f_y, grad_F_y = GetGradient(Y_sorted)
    # Since Y_sorted is sorted, the empirical CDF will be equally spaced in ascending order from 0 to 1
    F_y = torch.linspace(0, 1, Nsims + 1)[1:].reshape(-1, 1).to(device)
    
    # Operations outside of GetGradient() are using the no gradient copies of tensor X and Y
    rank_diff = Y_sorted_nograd - X_sorted_nograd
    wass_order, wass_limit = wass_params.GetParams()
    wass_dist = torch.mean(torch.abs(rank_diff) ** wass_order) ** (1 / wass_order)

    alpha, beta, p, rm_type = rm_params.GetParams()
    # RM_weight is the risk measure contribution to the gradient and depends on the $\gamma$ RM distortion function
    if rm_type == 'alpha-beta':
        # Equation 4.1 in the paper
        norm_factor = p * alpha + (1 - p) * (1 - beta)
        RM_weight = (p * (F_y <= alpha) + (1 - p) * (F_y > beta)) / norm_factor
    
    elif rm_type == 'mean-CVaR':
        RM_weight = 10 / 11 * (F_y <= alpha) / alpha + 1 / 11

    else:
        print("Risk Measure Type Not Supported.")
    
    # Find the full inner problem gradient using inner gradient formula in the paper (Equation 3.5)
    if problem_type == "inner":
        lam, mu, _ = lm_params.GetParams()
        constr_err = wass_dist ** wass_order - wass_limit ** wass_order
        Lambda = (lam + mu * constr_err) * (wass_dist > wass_limit)
        # LM_weight is the Wasserstein constraint contribution to the gradient
        LM_weight = wass_order * Lambda * torch.abs(rank_diff) ** (wass_order - 1) * torch.sign(rank_diff)

        # Loss depends on if the objective is to maximize or minimize RM
        if rm_objective == "minimize":
            total_weight = RM_weight - LM_weight

        elif rm_objective == "maximize":
            total_weight = -RM_weight - LM_weight
        
        else:
            print("Risk Measure Objective Not Supported.")
        
        # Only grad_F_y contains any gradients
        loss = torch.mean(grad_F_y * total_weight / f_y)
    
    # Find the full outer problem gradient using outer gradient formula in the paper (Equation 3.7)
    elif problem_type == "outer":
        loss = torch.mean(grad_F_y * RM_weight / f_y)

    rm_phi = GetRiskMeasure(X, rm_params)
    rm_theta = GetRiskMeasure(Y, rm_params)

    return loss, rm_phi.item(), rm_theta.item(), wass_dist.item()


def GetGradient(X):
    """
    Calculates the gradient of the distribution function of input X using the Kernel Density Estimation (KDE) approach.

    Parameters:
    X : torch.Tensor
        Input tensor for which the gradient of the distribution function is to be calculated.

    Returns:
    tuple
        A tuple containing:
        - f_x : torch.Tensor
            The estimated density function of X.
        - grad_F_x : torch.Tensor
            The gradient of the distribution function of X.
    """
    n = X.shape[0]
    X_no_grad = X.detach()
    normal = torch.distributions.Normal(0, 1)
    # Bandwidth size using a scaled down version of Silverman's rule
    h = 1.06 * torch.std(X_no_grad) * n ** (-1 / 5) / 2
    z_score = (X_no_grad.reshape(1, -1) - X_no_grad.reshape(-1, 1)) / h
    f_x = torch.mean(torch.exp(normal.log_prob(z_score)), axis=0) / h
    # Ensure gradients are only attached in calculation of grad_F_x through the X tensor
    grad_F_x = -torch.mean(torch.exp(normal.log_prob(z_score)) * X.reshape(-1, 1), axis=0) / h 

    return f_x.reshape(-1, 1), grad_F_x.reshape(-1, 1)

