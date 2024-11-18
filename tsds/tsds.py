import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from utils.lb import lb
from statsmodels.tsa.stattools import adfuller
from utils.trend import decompose
from utils.skew import skew
from utils.kurtosis import kurtosis
from utils.acf import acf

def get_stars(pvalue):
    if pvalue <= 0.01:
        return '***'
    elif pvalue <= 0.05:
        return '**'
    elif pvalue <= 0.1:
        return '*'
    return ''

def calculate_stats(ts, lb_lags=10, adf_regression='c'):
    """
    计算时间序列的统计量
    Args:
        ts: 时间序列数据
        lb_lags: Ljung-Box检验的滞后阶数
        adf_regression: ADF检验的回归类型 ('c'常数项, 'ct'常数项和趋势项, 'n'无常数项和趋势项)
    """
    mean = np.mean(ts)
    std = np.std(ts)
    skew_value = skew(ts)
    kurt_value = kurtosis(ts)
    
    # Ljung-Box test
    lb_stat, lb_pvalue = lb(ts, lb_lags)
    
    # ADF test
    adf_result = adfuller(ts, regression=adf_regression)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    
    return {
        'Mean': f"{mean:.4f}",
        'Std Dev': f"{std:.4f}",
        'Skewness': f"{skew_value:.4f}",
        'Kurtosis': f"{kurt_value:.4f}",
        f'LB({lb_lags})': f"{lb_stat:.4f}{get_stars(lb_pvalue)}",
        f'ADF({adf_regression})': f"{adf_stat:.4f}{get_stars(adf_pvalue)}"
    }

def ts_statistics(series_dict, save_path=False, lb_lags=10, adf_regression='c'):
    """
    打印时间序列的描述性统计表
    Args:
        series_dict: 包含时间序列数据的字典
        save_path: False表示不保存，True表示保存在当前目录，字符串表示保存在指定路径
        lb_lags: Ljung-Box检验的滞后阶数
        adf_regression: ADF检验的回归类型 ('c'常数项, 'ct'常数项和趋势项, 'n'无常数项和趋势项)
    """
    results = {}
    for name, ts in series_dict.items():
        results[name] = calculate_stats(ts, lb_lags=lb_lags, adf_regression=adf_regression)
    
    stats_df = pd.DataFrame(results).T
    
    print("\nDescriptive Statistics:")
    print(stats_df)
    print("\nNote: * p<0.1, ** p<0.05, *** p<0.01")
    
    # 处理保存逻辑
    if save_path:
        path = '.' if save_path is True else save_path
        stats_df.to_csv(f"{path}/time_series_statistics.csv")
    
    return stats_df

def plot_decomposition(series_dict, lag=10, period=12, save_path=False):
    """
    为每个时间序列绘制分解图
    Args:
        series_dict: 包含时间序列数据的字典
        lag: 自相关滞后阶数
        period: 季节性周期
        save_path: False表示不保存，True表示保存在当前目录，字符串表示保存在指定路径
    """
    for name, ts in series_dict.items():
        print(f"\n{name}:")
        
        # Decomposition
        decomposition = decompose(ts, period)
        trend = decomposition['trend']
        seasonal = decomposition['seasonal']
        residual = decomposition['resid']
        
        print(f"Trend component mean: {np.mean(trend[~np.isnan(trend)]):.4f}")
        print(f"Seasonal component amplitude: {np.ptp(seasonal[~np.isnan(seasonal)]):.4f}")
        
        # Autocorrelation
        acf_value = acf(ts, lag)
        print("Autocorrelations up to lag 10:")
        print([f"{x:.4f}" for x in acf_value[1:]])
        
        # Plot decomposition
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(ts)
        plt.title(f'{name} - Original')
        plt.subplot(412)
        plt.plot(trend)
        plt.title('Trend')
        plt.subplot(413)
        plt.plot(seasonal)
        plt.title('Seasonal')
        plt.subplot(414)
        plt.plot(residual)
        plt.title('Residual')
        plt.tight_layout()
        
        # 处理保存逻辑
        if save_path:
            # 如果save_path为True，使用当前目录
            path = '.' if save_path is True else save_path
            plt.savefig(f"{path}/{name}_decomposition.png")
        
        plt.show()




