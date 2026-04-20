import numpy as np
from scipy import signal

def calculate_energy_vector(data):
    """Calculates 3D magnitude, azimuth, and elevation."""
    # Handle both 4-channel (Ambisonic) and 2-channel (PCA components)
    if data.shape[1] >= 4:
        w, x, y, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        energy_w = np.sum(w**2) + 1e-10
        r_x, r_y, r_z = np.sum(x*w)/energy_w, np.sum(y*w)/energy_w, np.sum(z*w)/energy_w
    else:
        # Proxy for Compressed Azimuth using first two PCA components
        r_x, r_y, r_z = np.mean(data[:, 0]), np.mean(data[:, 1]), 0
        
    magnitude = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    azimuth = np.arctan2(r_y, r_x)
    elevation = np.arcsin(np.clip(r_z / (magnitude + 1e-10), -1, 1))
    return magnitude, azimuth, elevation

def get_3d_perceptual_metrics(orig, comp, recon, fs):
    """Calculates metrics across all three stages of the codec."""
    o_e = np.sqrt(np.sum(orig**2, axis=1))
    # Scale compressed energy to match original for visual comparison
    c_e = np.sqrt(np.sum(comp**2, axis=1)) * (np.max(o_e)/(np.max(comp)+1e-10))
    r_e = np.sqrt(np.sum(recon**2, axis=1))
    
    freqs, psd = signal.welch(o_e, fs, nperseg=1024)
    mask_thresh = psd * 0.05 + 1e-12 
    return o_e, c_e, r_e, freqs, psd, mask_thresh

def calculate_moving_cues(binaural_data, frame_size=1024):
    left, right = binaural_data[:, 0], binaural_data[:, 1]
    itds, ilds = [], []
    for i in range(0, len(left) - frame_size, frame_size):
        l_seg, r_seg = left[i:i+frame_size], right[i:i+frame_size]
        ild = 20 * np.log10((np.std(l_seg) + 1e-10) / (np.std(r_seg) + 1e-10))
        corr = signal.correlate(l_seg, r_seg, mode='same')
        itd = np.argmax(corr) - (frame_size // 2)
        itds.append(itd); ilds.append(ild)
    return np.array(itds), np.array(ilds)

def simple_binaural_render(data):
    left = (data[:, 0] + data[:, 2]) * 0.707
    right = (data[:, 0] - data[:, 2]) * 0.707
    return np.stack([left, right], axis=1)

def calculate_snr(orig, recon):
    noise = orig - recon
    return 10 * np.log10(np.sum(orig**2) / (np.sum(noise**2) + 1e-10))
