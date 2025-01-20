import numpy as np
# abstract base class to save and load parameters 
# requires a dictionary of parameters as inputs 
# GetParams() method returns a tuple of parameter values GetParams() 
class Params:
  def __init__(self, params):
    self.params = params
  def GetParams(self):
    raise Exception("Must be overridden in subclass.")

# 6 个主要参数类别
# MarketModel_Params 类包含与市场模型有关的所有参数，例如波动性
# Sim_Params 类包含与模拟有关的所有参数，例如模拟次数、时间范围
# LM_Params 类包含与增广拉格朗日方法有关的所有参数，例如拉格朗日乘数和惩罚约束
# LM_Params 还有一个额外的 UpdateParams() 方法，该方法接受 constr_err c[X^{\theta*}, X^{\phi}], /
# 并根据 $\lam \leftarrow \lam + \mu c[X^{\theta*}, X^{\phi}]$ 在每个 update_freq 时期更新拉格朗日乘数
# RM_params 类包含与风险衡量有关的所有参数，例如alpha、beta 和 p
# Wass_Params 类包含与 wasserstein 距离有关的所有参数，例如。/
# wass 约束 $d_p[X^{\theta}, X^{\phi}] \leq \epsilon$ 中的 wass 顺序和最大允许 wass 距离 $\epsilon$
# Train_Params 类包含与训练过程有关的所有参数，例如训练内部/外部问题的时期数

class MarketModel_Params(Params):
  
  def __init__(self, params, Type):
    Params.__init__(self, params)
    self.Type = Type

  def GetParams(self, measure = "P"):
    params = self.params
    if self.Type == "SIR_CEV":
        
      if measure == "P":
        mu = params["P"]["mu"]
        sigma = params["P"]["sigma"]
        beta = params["P"]["beta"]
        rho = params["P"]["rho"]
        r0 = params["P"]["r0"]
        kappa = params["P"]["kappa"]
        theta_r = params["P"]["theta_r"]
        sigma_r = params["P"]["sigma_r"] 
        
      elif measure == "Q":
        mu = params["P"]["mu"]
        sigma = params["P"]["sigma"]
        beta = params["P"]["beta"]
        rho = params["P"]["rho"]
        r0 = params["P"]["r0"]
        kappa = params["Q"]["kappa"]
        theta_r = params["Q"]["theta_r"]
        sigma_r = params["Q"]["sigma_r"] 

      return mu, sigma, beta, rho, r0, kappa, theta_r, sigma_r
    
    elif self.Type == "OU":
      sigma = params["sigma"]
      kappa = params["kappa"]
      theta = params["theta"]

      return sigma, kappa, theta

    elif self.Type == "Factor":
      pass
    
    else:
      print("Market Model Type Not Supported.")
          
class Sim_Params(Params):
  def __init__(self, params):
    Params.__init__(self, params)
    
  def GetParams(self):
    params = self.params
    Ndt = params.get("Ndt")
    T = params.get("T")
    Nsims = params.get("Nsims")
    Nassets = params.get("Nassets")
    # initial risky asset value (optional)
    S0 = params.get("S0")
    # initial portfolio value (optional)
    X0 = params.get("X0")
    # initial weights of X_phi strategy (optional)
    phi = params.get("phi")

    return Ndt, T, Nsims, Nassets, S0, X0, phi

class LM_Params(Params):
  def __init__(self, params, mu_cap = np.inf):
    Params.__init__(self, params)
    self.initial_lam = params["lam"]
    self.initial_mu = params["mu"]
    # maximum mu value allowed
    self.mu_cap = mu_cap
  
  def GetParams(self):
    params = self.params
    # lagrange multiplier
    lam = params["lam"]
    # penalty constraint
    mu = params["mu"]
    # mu_update is the $\alpha$ term in the penalty constraint update rule $\mu \leftarrow \alpha \mu$ 
    mu_update = params["mu_update"]
    # number of iterations before updating lam and mu
    update_freq = params["update_freq"]
    return lam, mu, update_freq
  
  def UpdateParams(self, constr_err):
    params = self.params
    # ensure lam is non-negative
    params["lam"] = max(params["lam"] + params["mu"]*constr_err, 0)
    # can set a cap on mu (optional)
    params["mu"] = min(params["mu"] * params["mu_update"], self.mu_cap)

    return params["lam"], params["mu"]

  # reset to initial params the class was initialized with
  def ResetParams(self):
    self.params["lam"] = self.initial_lam
    self.params["mu"] = self.initial_mu

    return self.params["lam"], self.params["mu"]

class RM_Params(Params):
  def __init__(self, params):
    Params.__init__(self, params)

  def GetParams(self):
    params = self.params
    alpha = params.get("alpha")
    beta = params.get("beta")
    p = params.get("p")
    rm_type = params["rm_type"]
    
    return alpha, beta, p, rm_type

class Wass_Params(Params):
  def __init__(self, params):
    Params.__init__(self, params)

  def GetParams(self):
    params = self.params
    # wass_order is the p in the p-order Wasserstein distance
    wass_order = params["wass_order"]
    # wass_limit is the maximum allowed Wasserstein distance 
    wass_limit = params["wass_limit"]

    return wass_order, wass_limit

class Train_Params(Params):
  def __init__(self, params):
    Params.__init__(self, params)

  def GetParams(self):
    params = self.params
    # number of epochs to train inner and outer networks
    inner_epochs = params["inner_epochs"]
    outer_epochs = params["outer_epochs"]
    # number of epochs before showing progress plots
    plot_freq_inner = params["plot_freq_inner"]
    plot_freq_outer = params["plot_freq_outer"]
    # learning rates of networks
    inner_lr = params["inner_lr"]
    outer_lr = params.get("outer_lr")
    # number of epochs to freeze market scenarios
    freeze_market_iter = params["freeze_market_iter"]
    # number of epochs to freeze inner network
    freeze_inner_iter = params["freeze_inner_iter"]

    return inner_epochs, outer_epochs, plot_freq_inner, plot_freq_outer, inner_lr, outer_lr, freeze_market_iter, freeze_inner_iter