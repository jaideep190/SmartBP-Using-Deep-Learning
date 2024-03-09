import numpy as np
from scipy.signal import resample

def main(data,):
    red_channel_signal = np.array(data)
    desired_samples=875
    resampled_signal = resample(red_channel_signal, 875)
    normalized_signal = (resampled_signal - np.mean(resampled_signal)) / np.std(resampled_signal)
    normalized_signal_input = normalized_signal.astype(np.float32)
    normalized_signal_input = normalized_signal_input.reshape((1, desired_samples, 1))
    return normalized_signal_input.flatten().tolist()
