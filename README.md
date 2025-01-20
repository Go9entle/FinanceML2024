# 期末报告代码及数据的说明文档

## 数据集说明

### 数据集来源

本期末报告里含有的数据集是通过Yahoo Finance收集10个资产

- 苹果公司（Apple Inc.）``AAPL`
- 微软公司（Microsoft Corp.）`MSFT`
- 谷歌母公司（Alphabet Inc.）`GOOGL`
- 亚马逊公司（Amazon.com Inc）`AMZN`
- 英伟达公司（NVIDIA Corp.）`NVDA`
- 债券iShares 20+ Year Treasury Bond20+年国债ETF `TLT`
- SPDR S&P 500 ETF Trust标普500ETF `SPY`
- SPDR Gold Shares 黄金ETF `GLD`
- WTI原油期货`CL=F`
- 比特币货币`BTC-USD`        

得到的。数据收集与处理代码在文件`dat_collect_processing.ipynb`中，主要运用包`yfinance`下载。以下是报告中数据集的解释

### 数据集解释

1. `stock_returns.csv`:10个资产从2015年1月1日到2025年1月1日10年间所有交易日的对数收益率（已去除NaN）。

2. `systematic_risk.csv`:10个资产从2015年1月1日到2025年1月1日10年间所有交易日系统性风险因子，对对数收益率矩阵用PCA分解后取第一主成分作为系统性风险因子，于是每个交易日都有一个特定的系统性风险因子。
3. `idiosyncratic_risk.csv`:对数收益率减去系统性风险因子后得到的特质性风险因子。
4. `final_simulated_returns.csv`:运用KDE核估计密度分布从系统性因子和特质性因子中抽取样本模拟资产路径。

## python代码文件说明

代码包括2个notebook文件`dat_collect_processing.ipynb`,`RobustPortfolioAllocation.ipynb`和4个python文件`parameters.py`,`metrics.py`,`market_models.py`,`solution.py`.

1. `dat_collect_processing.ipynb`:分为两个部分，一是数据的收集与处理，二是判断系统性风险因子是否能看作正态分布，利用QQ图以及核密度分布与正态分布比较得知，系统性因子的KDE具有尖峰厚尾的特点显然不能看作正态分布。
2. `RobustPortfolioAllocation.ipynb`:
   1. 是报告中案例的主要执行部分，构建强化学习网络，重载强化学习中的Agent类和Adversary类，赋值训练轮次等参数并对结果进行输出与可视化分析。
   2. 需要注意的是为了使得加快运行速度，用了pytorch和GPU加速运算，并减少了训练轮次。在本报告训练下大约每个训练模块只需要3分钟左右。
   3. 另外此文件包含几个模块，分别是前期的准备模块定义和重载一些参数和方法。定义好$\alpha-\beta$型的风险测度$\gamma(u)$后，变化Wasserstein距离扭曲最优投资组合并继续迭代优化。Wasserstein距离依照$\epsilon=10^{-3},10^{-2},10^{-1},1$由小增大，输出每次强化学习的过程结果与最终结果。
   4. 在最后两个模块进行了比较操作，随着Wasserstein距离的增大，最优投资组合各个资产权重的堆叠图以及最优投资组合终端时刻财富的分布并得到实证结果的结论。
3. `parameters.py`:定义参数类并从实例中获取参数传递给其他python文件。
4. `market_models.py`:定义市场模型类，实证主要运用基于真实数据模拟的因子模型。
5. `metrics.py`:定义风险测度以及计算梯度等操作。
6. `solution.py`:核心求解代码，主要定义了Agent类和Adversary类需要在实例中重载，构建内层训练与外层训练过程。