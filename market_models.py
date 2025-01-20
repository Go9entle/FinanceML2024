import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import gaussian_kde

class MarketModel:
  
  def __init__(self, params):
    self.params = params
  
  def Sim(self, sim_params):
    raise Exception("Must be overridden in subclass.")

  def PlotSim(self):
    S, t = self.S, self.t
    Nassets = S.shape[2]
    plt.figure(figsize=(5*Nassets, 5))
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=22)
    for i in range(Nassets):
      plt.subplot(1, Nassets, i+1)
      plt.fill_between(t, np.quantile(S[:,:,i], 0.1, axis=1).T, np.quantile(S[:,:,i], 0.9, axis=1).T, color='y', alpha=0.5)
      # plot first 100 paths
      plt.plot(t, S[:,:100,i], linewidth=0.3)
      # plot first path in a thicker line
      plt.plot(t, S[:,0,i], color='r', linewidth=1.5)
      # plot the 10th, 50th and 90th quantiles
      plt.plot(t, np.quantile(S[:,:,i],[0.1, 0.5, 0.9],axis=1).T, color='k', linewidth=1, linestyle='--')
      plt.xlabel("t")
      plt.ylabel("$S_t^" +str(i+1)+"$")
    
    plt.tight_layout()
    plt.show()

# 基于真实数据重写因子模型的实现
class Factor(MarketModel):
  """
  利用真实数据重写针对稳健资产配置的因子模型\n
  数据收集及处理详见文件dat_collect_processing.ipynb\n
  基本思路是得到近十年(2015-01-01~2025-01-01)对数收益率数据后，\n
  利用PCA得到第一主成分作为真实数据的系统性风险因子\n
  然后核密度估计法模拟得到Nsims条路径的系统性风险因子\n
  特质性风险因子由每天资产的对数收益率减去系统性风险因子得到\n
  同样利用多元核密度估计法模拟得到Nsims条路径的系统性风险因子
  """
  def __init__(self, params = None):
    MarketModel.__init__(self, params)
    self.t = 1
  
  def Sim(self, sim_params):
    _, _, Nsims, Nassets, _, _, _ = sim_params.GetParams()
    # Step 1:读取对数收益率数据，已经在dat_collect_processing.ipynb处理完成
    returns = pd.read_csv("stock_returns.csv",index_col=0)
    # Step 2: 使用PCA分解收益率
    pca = PCA()
    pca.fit(returns)

    # Step 3: 提取系统性风险因子
    sys_risk = returns @ pca.components_[0]  # 第一个主成分作为系统性风险因子
    # 确保 sys_risk 和 returns 是 NumPy 数组
    sys_risk = np.array(sys_risk)  # 转为 NumPy 数组
    returns_array = returns.values  # 提取返回值

    # 使用 NumPy 操作避免多维索引问题
    sys_risk_expanded = np.expand_dims(sys_risk, axis=1)

    # Step 4: 提取特质性风险因子
    idio_risk = returns_array - sys_risk_expanded @ pca.components_[:1]

    # 确保 sys_risk 和 idio_risk 是 NumPy 数组
    sys_risk = np.array(sys_risk).flatten()  # 展平成一维数组
    idio_risk = np.array(idio_risk)  # 确保是 NumPy 数组

    #Step 5: 利用真实数据KDE模拟系统性风险因子
    # 计算 KDE
    kde1 = gaussian_kde(sys_risk)
    kde2 = gaussian_kde(idio_risk.T)
    # 从 KDE 中随机取样
    simulated_sys_risk = kde1.resample(Nsims).flatten()
    simulated_idio_risk = kde2.resample(Nsims) .T


    # Step 6: 合并系统性风险因子和特质性风险因子
    # 这里假设每个资产的收益率是系统性风险因子和特质性风险因子的和
    final_simulated_returns = simulated_sys_risk[:, np.newaxis] + simulated_idio_risk
    # 将时间维度插入到S中以保持一致性
    self.S = np.expand_dims(final_simulated_returns, axis=0)

    return self.S
