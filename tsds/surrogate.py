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
    # Input validation
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if n_iterations < 1:
        raise ValueError("Number of iterations must be positive")
    if num_surrogates < 1:
        raise ValueError("Number of surrogates must be positive")
        
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

# 6. IAAWT Method

def dwt(signal, level=1):
    """
    Simple Discrete Wavelet Transform implementation using Haar wavelet.
    """
    result = []
    current = signal.copy()
    
    for _ in range(level):
        n = len(current)
        if n < 2:
            break
            
        # Compute approximation and detail coefficients
        h = current[:n-n%2].reshape(-1, 2)
        approx = (h[:, 0] + h[:, 1]) / np.sqrt(2)
        detail = (h[:, 0] - h[:, 1]) / np.sqrt(2)
        
        result.append(detail)
        current = approx
    
    result.append(current)
    return result[::-1]  # Return with approximation coefficients first

def idwt(coeffs):
    """
    Simple Inverse Discrete Wavelet Transform implementation using Haar wavelet.
    """
    current = coeffs[0]
    
    for detail in coeffs[1:]:
        # Upsampling
        n = len(current)
        expanded = np.zeros(2 * n)
        
        # Reconstruction
        expanded[::2] = (current + detail) / np.sqrt(2)
        expanded[1::2] = (current - detail) / np.sqrt(2)
        current = expanded
    
    return current

def iaawt(signal, n_iterations=1000, num_surrogates=1, level=None):
    """
    Generate surrogates using the Iterative Amplitude Adjusted Wavelet Transform (IAAWT) method.
    Uses a simple Haar wavelet implementation with numpy.
    
    Parameters
    ----------
    signal : array-like
        Input time series (1D array)
    n_iterations : int, optional
        Number of iterations (default: 1000)
    num_surrogates : int, optional
        Number of surrogate time series to generate (default: 1)
    level : int, optional
        Decomposition level. If None, it will be automatically determined
        based on the signal length.
        
    Returns
    -------
    list
        List of surrogate time series
    """
    # Input validation
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if n_iterations < 1:
        raise ValueError("Number of iterations must be positive")
    if num_surrogates < 1:
        raise ValueError("Number of surrogates must be positive")
    
    # Determine decomposition level if not specified
    if level is None:
        level = int(np.log2(len(signal))) - 1
    
    surrogates = []
    signal_length = len(signal)
    sorted_signal = np.sort(signal)
    
    # Get original wavelet coefficients
    coeffs_orig = dwt(signal, level=level)
    
    for _ in range(num_surrogates):
        # Initialize with random permutation
        surrogate = np.random.permutation(signal)
        
        for _ in range(n_iterations):
            # Wavelet decomposition of surrogate
            coeffs = dwt(surrogate, level=level)
            
            # Adjust coefficients
            new_coeffs = []
            for orig_c, curr_c in zip(coeffs_orig, coeffs):
                # Preserve amplitudes of original, but use phases of surrogate
                amp_orig = np.abs(orig_c)
                phases = np.angle(curr_c + 1j * np.finfo(float).eps)  # Add small constant to avoid division by zero
                new_c = amp_orig * np.exp(1j * phases)
                new_coeffs.append(new_c.real)
            
            # Inverse wavelet transform
            surrogate_temp = idwt(new_coeffs)
            
            # Amplitude adjustment
            if len(surrogate_temp) > signal_length:
                surrogate_temp = surrogate_temp[:signal_length]
            ranks = np.argsort(np.argsort(surrogate_temp))
            surrogate = sorted_signal[ranks]
        
        surrogates.append(surrogate)
    
    return surrogates

# 7. Shuffle Surrogate Method

def shuffle_surrogate(signal, num_surrogates=1, preserve_distribution=True):
    """
    Generate surrogates using the Shuffle Surrogate method by randomly permuting the original time series.
    
    Parameters
    ----------
    signal : array-like
        Input time series (1D array)
    num_surrogates : int, optional
        Number of surrogate time series to generate (default: 1)
    preserve_distribution : bool, optional
        If True, ensures exact preservation of the original amplitude distribution.
        If False, allows for small variations due to random sampling (default: True)
        
    Returns
    -------
    list
        List of surrogate time series
    
    Notes
    -----
    This method completely destroys both linear and nonlinear correlations while
    preserving the amplitude distribution of the original signal.
    """
    # Input validation
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if num_surrogates < 1:
        raise ValueError("Number of surrogates must be positive")
    
    surrogates = []
    
    if preserve_distribution:
        # Generate surrogates using permutation
        for _ in range(num_surrogates):
            surrogate = np.random.permutation(signal)
            surrogates.append(surrogate)
    else:
        # Generate surrogates using bootstrap resampling
        signal_length = len(signal)
        for _ in range(num_surrogates):
            indices = np.random.randint(0, signal_length, size=signal_length)
            surrogate = signal[indices]
            surrogates.append(surrogate)
    
    return surrogates

# 8. MBB Surrogate Method

def mbb_surrogate(signal, block_length=None, num_surrogates=1, overlap=True):
    """
    Generate surrogates using the Moving Block Bootstrap method.
    
    Parameters
    ----------
    signal : array-like
        Input time series (1D array)
    block_length : int, optional
        Length of blocks to use. If None, will be automatically determined
        as sqrt(signal length) rounded to nearest integer.
    num_surrogates : int, optional
        Number of surrogate time series to generate (default: 1)
    overlap : bool, optional
        Whether to use overlapping blocks (default: True)
        
    Returns
    -------
    list
        List of surrogate time series
    
    Notes
    -----
    This method preserves the local temporal structure within blocks while
    randomizing the relationships between blocks. The block_length parameter
    controls the trade-off between preserving local structure (larger blocks)
    and maintaining variability (smaller blocks).
    """
    # Input validation
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if num_surrogates < 1:
        raise ValueError("Number of surrogates must be positive")
    
    signal_length = len(signal)
    
    # Determine block length if not specified
    if block_length is None:
        block_length = int(np.sqrt(signal_length))
    elif block_length < 1:
        raise ValueError("Block length must be positive")
    elif block_length > signal_length:
        raise ValueError("Block length cannot be larger than signal length")
    
    surrogates = []
    
    # Calculate number of blocks needed
    if overlap:
        # For overlapping blocks
        available_blocks = signal_length - block_length + 1
        blocks_needed = int(np.ceil(signal_length / block_length))
    else:
        # For non-overlapping blocks
        available_blocks = signal_length // block_length
        blocks_needed = available_blocks
    
    for _ in range(num_surrogates):
        surrogate = np.zeros(signal_length)
        current_position = 0
        
        # Generate surrogate by concatenating random blocks
        while current_position < signal_length:
            # Randomly select block start position
            if overlap:
                block_start = np.random.randint(0, signal_length - block_length + 1)
            else:
                block_start = np.random.randint(0, available_blocks) * block_length
            
            # Extract block
            block = signal[block_start:block_start + block_length]
            
            # Add block to surrogate
            end_position = min(current_position + block_length, signal_length)
            block_portion = end_position - current_position
            surrogate[current_position:end_position] = block[:block_portion]
            
            current_position = end_position
        
        surrogates.append(surrogate)
    
    return surrogates

# 9. Phase Randomization Method

def phase_randomization(signal, num_surrogates=1, preserve_symmetry=True):
    """
    Generate surrogates using the Phase Randomization method.
    
    Parameters
    ----------
    signal : array-like
        Input time series (1D array)
    num_surrogates : int, optional
        Number of surrogate time series to generate (default: 1)
    preserve_symmetry : bool, optional
        If True, ensures that the phase randomization preserves
        the Hermitian symmetry of the Fourier transform for real signals
        (default: True)
        
    Returns
    -------
    list
        List of surrogate time series
    
    Notes
    -----
    This method preserves the linear correlation structure (power spectrum) 
    while destroying nonlinear dependencies by randomizing the Fourier phases.
    The amplitude spectrum of the original signal is exactly preserved.
    """
    # Input validation
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if num_surrogates < 1:
        raise ValueError("Number of surrogates must be positive")
    
    signal_length = len(signal)
    surrogates = []
    
    # Get original Fourier transform
    fft_signal = np.fft.fft(signal)
    amplitudes = np.abs(fft_signal)
    
    # Generate surrogates
    for _ in range(num_surrogates):
        if preserve_symmetry:
            # Generate random phases preserving Hermitian symmetry
            random_phases = np.random.uniform(0, 2*np.pi, size=(signal_length // 2 + 1))
            phases = np.zeros(signal_length)
            phases[0] = 0  # DC component remains unchanged
            phases[1:signal_length//2 + 1] = random_phases[1:]
            phases[signal_length//2 + 1:] = -random_phases[1:signal_length//2][::-1]
            
            # Special handling for even-length signals
            if signal_length % 2 == 0:
                phases[signal_length//2] = 0  # Nyquist frequency component remains real
        else:
            # Simple random phases without symmetry constraint
            phases = np.random.uniform(0, 2*np.pi, size=signal_length)
            phases[0] = 0  # Keep DC component real
        
        # Generate surrogate in frequency domain
        surrogate_fft = amplitudes * np.exp(1j * phases)
        
        # Transform back to time domain
        surrogate = np.real(np.fft.ifft(surrogate_fft))
        
        surrogates.append(surrogate)
    
    return surrogates

