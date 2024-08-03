import numpy as np

def moving_window_average(data, window_size):
    half_window = window_size // 2
    result = np.empty_like(data)
    for i in range(data.size):
        start = max(0, i - half_window)
        end = min(data.size, i + half_window + 1)
        result[i] = np.mean(data[start:end])
    return result