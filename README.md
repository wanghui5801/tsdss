<div align="center">
  <img src="https://ice.frostsky.com/2024/11/20/3b10b65c851c89af8ac5e11a72db1244.webp" alt="TSDSS Logo" width="200"/>
</div>

# TSDSS 📊 🔮 📈

[![PyPI version](https://badge.fury.io/py/tsdss.svg)](https://badge.fury.io/py/tsdss)
[![Python](https://img.shields.io/pypi/pyversions/tsdss.svg)](https://pypi.org/project/tsdss/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/tsdss)](https://pepy.tech/project/tsdss)
[![Build Status](https://github.com/wanghui5801/tsdss/workflows/Python%20Tests/badge.svg)](https://github.com/wanghui5801/tsdss/actions)


TSDSS is a comprehensive Python package for time series analysis and surrogate data generation. It provides a wide range of tools for statistical analysis, preprocessing, feature extraction, and surrogate data generation for both univariate and multivariate time series.

## Features

### Time Series Analysis
- Basic statistics (mean, std, skewness, kurtosis)
- Stationarity tests (ADF test, Ljung-Box test)
- Correlation analysis (Pearson, Spearman, Kendall)
- Spectral analysis
- Nonlinear analysis (Lyapunov exponent, phase space reconstruction)
- Entropy measures

### Time Series Preprocessing
- Missing value interpolation
- Outlier detection
- Normalization
- Resampling
- Feature extraction

### Surrogate Data Generation
- IAAFT (Iterative Amplitude Adjusted Fourier Transform)
- IAAFT+ (Enhanced IAAFT)
- IPFT (Iterative Phase-adjusted Fourier Transform)
- AIAAFT (Adaptive IAAFT)
- IAAWT (Iterative Amplitude Adjusted Wavelet Transform)
- Multivariate surrogate methods
- Bootstrap methods

### Time Series Filtering

Each filter has its own characteristics and use cases:

- **Moving Average Filter**: Simple and effective for reducing random noise
- **Exponential Filter**: Gives more weight to recent data points
- **Savitzky-Golay Filter**: Preserves higher moments of the data while smoothing
- **Kalman Filter**: Optimal for tracking time-varying signals
- **Butterworth Filter**: Frequency domain filtering with flat response
- **Median Filter**: Excellent for removing impulse noise and outliers

For multivariate time series, the `multivariate_filter` function provides a unified interface to apply any of these filters to each dimension of the data. Key features:

- Supports all single-variable filtering methods
- Maintains correlations between dimensions
- Handles errors gracefully for each dimension
- Preserves the original data structure

## Installation

```bash
pip install tsdss
```

## Input Data Format

TSDSS accepts the following input formats:
- NumPy arrays (1D for univariate, 2D for multivariate)
- Pandas Series (for univariate)
- Pandas DataFrame (for multivariate)

Example shapes:
- Univariate: (n_samples,) or (n_samples, 1)
- Multivariate: (n_samples, n_features)

## Quick Start Examples

### Basic Statistics and Analysis

```python
import numpy as np
import pandas as pd
from tsdss  import ts_statistics, plot_decomposition, calculate_entropy

# Basic time series statistics
ts = np.random.normal(0, 1, 1000)
stats = ts_statistics(ts)
print(stats)

# Plot time series decomposition
plot_decomposition(ts)

# Calculate entropy
entropy = calculate_entropy(ts)
print(f"Entropy: {entropy}")
```

### Time Series Preprocessing

```python
from tsdss import interpolate_missing, detect_outliers, normalize_ts, resample_ts

# Handle missing values
ts = pd.Series([1, np.nan, 3, np.nan, 5])
ts_clean = interpolate_missing(ts, method='linear')  # Options: linear, ffill, bfill, cubic, spline

# Detect outliers
ts = np.random.normal(0, 1, 1000)
outliers = detect_outliers(ts, method='zscore', threshold=3)  # Options: zscore, iqr, mad

# Normalize data
ts_norm = normalize_ts(ts, method='zscore')  # Options: zscore, minmax, robust

# Resample time series (requires datetime index)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)
ts_resampled = resample_ts(ts, freq='W', method='mean')
```

### Feature Extraction

```python
from tsdss import extract_time_features, extract_freq_features

# Extract time domain features
ts = np.random.normal(0, 1, 1000)
time_features = extract_time_features(ts)
print("Time domain features:", time_features)

# Extract frequency domain features
freq_features = extract_freq_features(ts)
print("Frequency domain features:", freq_features)
```

### Correlation Analysis

```python
from tsdss import mutual_information, kendall_correlation

# Calculate mutual information
x = np.random.normal(0, 1, 1000)
y = 0.5 * x + np.random.normal(0, 1, 1000)
mi = mutual_information(x, y)
print(f"Mutual Information: {mi}")

# Calculate Kendall correlation
kendall = kendall_correlation(x, y)
print(f"Kendall Correlation: {kendall}")
```

### Surrogate Data Generation

```python
from tsdss  import (
    iaaft, iaaft_plus, ipft, aiaaft, 
    multivariate_iaaft, block_bootstrap, 
    stationary_bootstrap
)

# Generate univariate surrogate data
ts = np.random.normal(0, 1, 1000)

# IAAFT method
surrogate_iaaft = iaaft(ts, n_iterations=1000, num_surrogates=1)[0]

# IAAFT+ method
surrogate_iaaft_plus = iaaft_plus(ts, n_iterations=1000, num_surrogates=1)[0]

# IPFT method
surrogate_ipft = ipft(ts, n_iterations=1000, num_surrogates=1)[0]

# Generate multivariate surrogate data
data = np.random.normal(0, 1, (1000, 3))  # 3-dimensional time series
mv_surrogate = multivariate_iaaft(data, max_iter=100, num_surrogates=1)[0]

# Bootstrap methods
block_samples = block_bootstrap(ts, block_length=50, num_bootstrap=100)
stat_samples = stationary_bootstrap(ts, mean_block_length=50, num_bootstrap=100)
```

### Wavelet Analysis

```python  
from tsdss import dwt, idwt, iaawt

# Perform discrete wavelet transform
ts = np.random.normal(0, 1, 1024)  # Length should be power of 2
coeffs = dwt(ts, level=3)

# Perform inverse wavelet transform
reconstructed = idwt(coeffs)

# Generate wavelet-based surrogate
surrogate = iaawt(ts, n_iterations=1000, num_surrogates=1)[0]
```

### Advanced Multivariate Analysis

```python
from tsdss import (
    mvts_surrogate_s_transform, 
    mvts_surrogate_wavelet,
    mvts_surrogate_pca,
    copula_surrogate
)

# Generate multivariate data
data = np.random.normal(0, 1, (1000, 5))

# Different multivariate surrogate methods
surrogate_st = mvts_surrogate_s_transform(data, num_surrogates=1)[0]
surrogate_wavelet = mvts_surrogate_wavelet(data, num_surrogates=1)[0]
surrogate_pca = mvts_surrogate_pca(data, num_surrogates=1)[0]
surrogate_copula = copula_surrogate(data, num_surrogates=1)[0]
```

### Bootstrap Methods

```python
from tsdss import block_bootstrap, stationary_bootstrap

# 1. Block Bootstrap
# Fixed block length, suitable for data with strong local dependencies
ts = np.random.normal(0, 1, 1000)
block_samples = block_bootstrap(
    data=ts, 
    block_length=50,  # Fixed block length
    num_bootstrap=100
)

# 2. Stationary Bootstrap
# Random block length (geometric distribution), preserves stationarity
stat_samples = stationary_bootstrap(
    data=ts, 
    mean_block_length=50,  # Average block length
    num_bootstrap=100
)

# Compare the two methods
print("Block Bootstrap first sample:", block_samples[0][:10])
print("Stationary Bootstrap first sample:", stat_samples[0][:10])

# Using with pandas Series
ts_series = pd.Series(ts)
block_samples_pd = block_bootstrap(ts_series, block_length=50, num_bootstrap=100)
stat_samples_pd = stationary_bootstrap(ts_series, mean_block_length=50, num_bootstrap=100)

# Key differences:
# 1. Block Bootstrap: Uses fixed block length
# 2. Stationary Bootstrap: Uses random block length (geometric distribution)
#    - Better preserves stationarity
#    - More suitable for time series with varying dependence structures
```

### Time Series Filtering

```python
from tsdss import (
    moving_average_filter,
    exponential_filter,
    savitzky_golay_filter,
    kalman_filter,
    butterworth_filter,
    median_filter,
    multivariate_filter
)
import numpy as np
import matplotlib.pyplot as plt

# 1. Univariate Filtering Example
t = np.linspace(0, 10, 1000)
noisy_signal = np.sin(2*np.pi*0.5*t) + 0.5*np.random.normal(0, 1, 1000)

# Apply different filters
ma_filtered = moving_average_filter(noisy_signal, window_size=5)
ema_filtered = exponential_filter(noisy_signal, alpha=0.3)
sg_filtered = savitzky_golay_filter(noisy_signal, window_size=15, poly_order=3)
kalman_filtered = kalman_filter(noisy_signal, Q=1e-5, R=1e-2)

# 2. Multivariate Filtering Example
# Generate sample multivariate data
mv_data = np.column_stack([
    np.sin(2*np.pi*0.5*t) + 0.5*np.random.normal(0, 1, 1000),
    np.cos(2*np.pi*0.3*t) + 0.3*np.random.normal(0, 1, 1000),
    0.5*t + np.random.normal(0, 0.2, 1000)
])

# Apply multivariate filter
mv_filtered = multivariate_filter(
    mv_data,
    filter_type='kalman',
    Q=1e-5,
    R=1e-2
)

# You can also try different filter types
mv_ma = multivariate_filter(mv_data, filter_type='ma', window_size=5)
mv_butter = multivariate_filter(
    mv_data, 
    filter_type='butter',
    cutoff=0.1,
    fs=100
)
```

## Performance

The package uses optimized C++ implementations for core computations:
- Trend decomposition
- Skewness and kurtosis calculation
- ACF computation
- Ljung-Box test

## Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- Pandas >= 1.0.0
- SciPy >= 1.6.0
- Statsmodels >= 0.13.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Version History

### 0.2.0
- Added comprehensive time series filtering functionality
- Added multivariate filtering support
- Improved documentation and examples
- Bug fixes and performance improvements

### 0.1.0
- Initial release
