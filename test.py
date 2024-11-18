# 1. IAAFT Method
import numpy as np

def iaaft(signal, n_iterations=1000, num_surrogates=1):
    """
    IAAFT surrogate generation method.
    Args:
        signal (array-like): Original time series.
        n_iterations (int): Number of iterations to perform.
        num_surrogates (int): Number of surrogate time series to generate.
    Returns:
        list: List of surrogate time series.
    """
    surrogates = []
    for _ in range(num_surrogates):
        original_spectrum = np.fft.fft(signal)
        sorted_signal = np.sort(signal)
        surrogate = np.random.permutation(signal)

        for _ in range(n_iterations):
            phase_adjusted = np.fft.ifft(original_spectrum * np.exp(1j * np.angle(np.fft.fft(surrogate))))
            surrogate = np.real(phase_adjusted)
            surrogate = sorted_signal[np.argsort(np.argsort(surrogate))]
        
        surrogates.append(surrogate)
    return surrogates

# 2. Enhanced IAAFT Method (IAAFT+)

def iaaft_plus(signal, n_iterations=2000, num_surrogates=1):
    """
    Enhanced IAAFT (IAAFT+) surrogate generation method.
    Args:
        signal (array-like): Original time series.
        n_iterations (int): Number of iterations to perform.
        num_surrogates (int): Number of surrogate time series to generate.
    Returns:
        list: List of surrogate time series.
    """
    surrogates = []
    for _ in range(num_surrogates):
        original_spectrum = np.fft.fft(signal)
        sorted_signal = np.sort(signal)
        surrogate = np.random.permutation(signal)

        for _ in range(n_iterations):
            phase_adjusted = np.fft.ifft(original_spectrum * np.exp(1j * np.angle(np.fft.fft(surrogate))))
            surrogate = np.real(phase_adjusted)
            surrogate = sorted_signal[np.argsort(np.argsort(surrogate))]
            # Additional enhancement iteration
            original_spectrum = np.fft.fft(surrogate)
        
        surrogates.append(surrogate)
    return surrogates

# 3. IPFT Method

def ipft(signal, n_iterations=1000, num_surrogates=1):
    """
    Iteratively Phase Adjusted Fourier Transform method.
    Args:
        signal (array-like): Original time series.
        n_iterations (int): Number of iterations to perform.
        num_surrogates (int): Number of surrogate time series to generate.
    Returns:
        list: List of surrogate time series.
    """
    surrogates = []
    for _ in range(num_surrogates):
        original_spectrum = np.fft.fft(signal)
        surrogate = np.random.permutation(signal)
        
        for _ in range(n_iterations):
            phase_adjusted = np.fft.ifft(original_spectrum * np.exp(1j * np.angle(np.fft.fft(surrogate))))
            surrogate = np.real(phase_adjusted)
        
        surrogates.append(surrogate)
    return surrogates

# 4. ISAAFT Method

def isaaft(signal, n_iterations=1000, num_surrogates=1):
    """
    Iteratively Shifted Amplitude Adjusted Fourier Transform method.
    Args:
        signal (array-like): Original time series.
        n_iterations (int): Number of iterations to perform.
        num_surrogates (int): Number of surrogate time series to generate.
    Returns:
        list: List of surrogate time series.
    """
    surrogates = []
    for _ in range(num_surrogates):
        original_spectrum = np.fft.fft(signal)
        sorted_signal = np.sort(signal)
        surrogate = np.random.permutation(signal)

        for _ in range(n_iterations):
            shifted_spectrum = np.roll(original_spectrum, np.random.randint(len(original_spectrum)))
            phase_adjusted = np.fft.ifft(shifted_spectrum * np.exp(1j * np.angle(np.fft.fft(surrogate))))
            surrogate = np.real(phase_adjusted)
            surrogate = sorted_signal[np.argsort(np.argsort(surrogate))]
        
        surrogates.append(surrogate)
    return surrogates

# 5. AIAAFT Method

def aiaaft(signal, n_iterations=1000, num_surrogates=1):
    """
    Adaptive Iteratively Amplitude Adjusted Fourier Transform method.
    Args:
        signal (array-like): Original time series.
        n_iterations (int): Number of iterations to perform.
        num_surrogates (int): Number of surrogate time series to generate.
    Returns:
        list: List of surrogate time series.
    """
    surrogates = []
    for _ in range(num_surrogates):
        original_spectrum = np.fft.fft(signal)
        sorted_signal = np.sort(signal)
        surrogate = np.random.permutation(signal)

        for _ in range(n_iterations):
            adaptive_spectrum = original_spectrum * (1 + np.random.uniform(-0.05, 0.05, len(original_spectrum)))
            phase_adjusted = np.fft.ifft(adaptive_spectrum * np.exp(1j * np.angle(np.fft.fft(surrogate))))
            surrogate = np.real(phase_adjusted)
            surrogate = sorted_signal[np.argsort(np.argsort(surrogate))]
        
        surrogates.append(surrogate)
    return surrogates

# Example: How to use these functions
if __name__ == "__main__":
    # Create a sample time series
    time_series = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * np.random.randn(100)
    
    # Generate 3 surrogates using each method
    surrogate_iaaft = iaaft(time_series, num_surrogates=3)
    surrogate_iaaft_plus = iaaft_plus(time_series, num_surrogates=3)
    surrogate_ipft = ipft(time_series, num_surrogates=3)
    surrogate_isaaft = isaaft(time_series, num_surrogates=3)
    surrogate_aiaaft = aiaaft(time_series, num_surrogates=3)
    
    # Print the surrogate arrays
    print("IAAFT Surrogate:", surrogate_iaaft)
    print("IAAFT+ Surrogate:", surrogate_iaaft_plus)
    print("IPFT Surrogate:", surrogate_ipft)
    print("ISAAFT Surrogate:", surrogate_isaaft)
    print("AIAAFT Surrogate:", surrogate_aiaaft)

