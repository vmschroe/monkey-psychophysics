import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats

def analyze_time_series_correlation(timestamps1, values1, timestamps2, values2):
    """
    Robust cross-correlation analysis with error handling
    
    Parameters:
    -----------
    timestamps1 : array-like
        Timestamps for first series
    values1 : array-like
        Values for first series
    timestamps2 : array-like
        Timestamps for second series
    values2 : array-like
        Values for second series
    
    Returns:
    --------
    dict : Correlation analysis results
    """
    # Ensure inputs are numpy arrays
    ts1 = np.array(timestamps1)
    val1 = np.array(values1, dtype=float)
    ts2 = np.array(timestamps2)
    val2 = np.array(values2, dtype=float)
    
    # Check for constant or zero-variation series
    if np.std(val1) == 0 or np.std(val2) == 0:
        print("Warning: One or both series have no variation!")
        return {
            'correlation': None,
            'p_value': None,
            'error': 'Constant or zero-variation series'
        }
    
    # Create a common time grid
    min_time = max(ts1.min(), ts2.min())
    max_time = min(ts1.max(), ts2.max())
    
    # Handle case with insufficient time range
    if min_time >= max_time:
        print("Warning: Insufficient time overlap!")
        return {
            'correlation': None,
            'p_value': None,
            'error': 'Insufficient time overlap'
        }
    
    # Interpolate series
    common_timestamps = np.linspace(min_time, max_time, 200)
    
    try:
        # Interpolation with error handling
        interp_func1 = interpolate.interp1d(ts1, val1, kind='linear', fill_value='extrapolate')
        interp_func2 = interpolate.interp1d(ts2, val2, kind='linear', fill_value='extrapolate')
        
        interpolated_series1 = interp_func1(common_timestamps)
        interpolated_series2 = interp_func2(common_timestamps)
        
        # Compute correlations
        pearson_corr, p_value = stats.pearsonr(interpolated_series1, interpolated_series2)
        
        # Safe cross-correlation
        centered_series1 = interpolated_series1 - interpolated_series1.mean()
        centered_series2 = interpolated_series2 - interpolated_series2.mean()
        
        # Avoid division by zero
        std1 = np.std(interpolated_series1)
        std2 = np.std(interpolated_series2)
        
        cross_corr = np.correlate(centered_series1, centered_series2, mode='full')
        
        # Normalize cross-correlation safely
        if std1 * std2 != 0:
            cross_corr /= (std1 * std2 * len(interpolated_series1))
        
        # Find max correlation and lag
        max_corr_index = np.argmax(np.abs(cross_corr))
        max_correlation = cross_corr[max_corr_index]
        lag = max_corr_index - len(interpolated_series1) + 1
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Original Series
        plt.subplot(1, 3, 1)
        plt.scatter(ts1, val1, label='Series 1', alpha=0.7)
        plt.scatter(ts2, val2, label='Series 2', alpha=0.7)
        plt.title('Original Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        
        # Interpolated Series
        plt.subplot(1, 3, 2)
        plt.plot(common_timestamps, interpolated_series1, label='Series 1')
        plt.plot(common_timestamps, interpolated_series2, label='Series 2')
        plt.title('Interpolated Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        
        # Cross-Correlation
        plt.subplot(1, 3, 3)
        plt.plot(cross_corr)
        plt.title('Cross-Correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'correlation': pearson_corr,
            'p_value': p_value,
            'max_cross_correlation': max_correlation,
            'lag_at_max_correlation': lag
        }
    
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return {
            'correlation': None,
            'p_value': None,
            'error': str(e)
        }
    
def diagnose_time_series(timestamps1, values1, timestamps2, values2):
    """
    Comprehensive diagnostic for time series correlation issues
    
    Parameters:
    -----------
    timestamps1 : array-like
        Timestamps for first series
    values1 : array-like
        Values for first series
    timestamps2 : array-like
        Timestamps for second series
    values2 : array-like
        Values for second series
    
    Returns:
    --------
    dict : Diagnostic information
    """
    # Convert to numpy arrays
    ts1 = np.array(timestamps1)
    val1 = np.array(values1, dtype=float)
    ts2 = np.array(timestamps2)
    val2 = np.array(values2, dtype=float)
    
    # Diagnostic checks
    diagnostics = {
        'series1_details': {
            'length': len(val1),
            'unique_values': len(np.unique(val1)),
            'min': np.min(val1),
            'max': np.max(val1),
            'mean': np.mean(val1),
            'std': np.std(val1),
            'has_nan': np.isnan(val1).any(),
            'has_inf': np.isinf(val1).any(),
        },
        'series2_details': {
            'length': len(val2),
            'unique_values': len(np.unique(val2)),
            'min': np.min(val2),
            'max': np.max(val2),
            'mean': np.mean(val2),
            'std': np.std(val2),
            'has_nan': np.isnan(val2).any(),
            'has_inf': np.isinf(val2).any(),
        },
        'timestamp_details': {
            'min_ts1': np.min(ts1),
            'max_ts1': np.max(ts1),
            'min_ts2': np.min(ts2),
            'max_ts2': np.max(ts2),
            'time_overlap': max(0, min(np.max(ts1), np.max(ts2)) - max(np.min(ts1), np.min(ts2)))
        }
    }
    
    # Correlation attempts
    try:
        # Pearson correlation
        corr, p_value = stats.pearsonr(val1, val2)
        diagnostics['pearson_correlation'] = {
            'correlation': corr,
            'p_value': p_value
        }
    except Exception as e:
        diagnostics['pearson_correlation_error'] = str(e)
    
    # Spearman correlation
    try:
        spearman_corr, spearman_p = stats.spearmanr(val1, val2)
        diagnostics['spearman_correlation'] = {
            'correlation': spearman_corr,
            'p_value': spearman_p
        }
    except Exception as e:
        diagnostics['spearman_correlation_error'] = str(e)
    
    return diagnostics

def print_diagnostics(diagnostics):
    """
    Pretty print the diagnostics
    """
    print("\n--- Series 1 Details ---")
    for key, value in diagnostics['series1_details'].items():
        print(f"{key}: {value}")
    
    print("\n--- Series 2 Details ---")
    for key, value in diagnostics['series2_details'].items():
        print(f"{key}: {value}")
    
    print("\n--- Timestamp Details ---")
    for key, value in diagnostics['timestamp_details'].items():
        print(f"{key}: {value}")
    
    # Print correlation results
    if 'pearson_correlation' in diagnostics:
        print("\n--- Pearson Correlation ---")
        for key, value in diagnostics['pearson_correlation'].items():
            print(f"{key}: {value}")
    
    if 'spearman_correlation' in diagnostics:
        print("\n--- Spearman Correlation ---")
        for key, value in diagnostics['spearman_correlation'].items():
            print(f"{key}: {value}")

def main():
    
    timestamps1 = times
    timestamps2 = times
    
    # Create a numerical series with some pattern
    values1 = rewards
    
    # Create a boolean series with some relationship to the numerical series
    values2 = correct
    
    results1 = analyze_time_series_correlation(timestamps1, values1, timestamps2, values2)
    print(results1)

if __name__ == "__main__":
    main()