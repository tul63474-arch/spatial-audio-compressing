import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import time
import io
import colour
from compressor import SpatialCodec
import spatial_utils as utils

st.set_page_config(page_title="Spatial Audio Analyzer", layout="wide")
colour.apply_custom_css()

with st.sidebar:
    st.title("Codec Control")
    uploaded_file = st.file_uploader("Upload file Ambisonic (.wav)", type=["wav"])
    st.divider()
    # Dynamic slider based on input channel count
    if uploaded_file is not None:
        temp_data, _ = sf.read(uploaded_file)
        uploaded_file.seek(0)
        num_channels = temp_data.shape[1]
    else:
        num_channels = 4
    
    n_comp = st.slider("Number of PCA Components", 1, num_channels, 2 if num_channels >= 2 else 1)

@st.cache_data
def process_audio_data(file_input, n_components):
    data, fs = sf.read(file_input)
    # Validate FOA (First-Order Ambisonics) requirement
    if data.shape[1] < 4:
        return None, "Error: File has to have 4 channels Ambisonic (WXYZ)."
    # Execute compression-decompression cycle
    codec = SpatialCodec(n_components=n_components)
    start_time = time.time()
    compressed = codec.compress(data)
    reconstructed = codec.decompress(compressed)
    latency = (time.time() - start_time) * 1000
    # Spatial Rendering & Metric Extraction
    binaural = utils.simple_binaural_render(reconstructed)
    o_e, c_e, r_e, freqs, psd, thresh = utils.get_3d_perceptual_metrics(data, compressed, reconstructed, fs)
    itds, ilds = utils.calculate_moving_cues(binaural)
    # Localization Analysis (Energy Vectors)
    snr = utils.calculate_snr(data, reconstructed)
    mag_o, az_o, el_o = utils.calculate_energy_vector(data)
    mag_r, az_r, el_r = utils.calculate_energy_vector(reconstructed)

    results_dict = {
        "data": data, "recon": reconstructed, "bin": binaural, "fs": fs,
        "snr": snr, "lat": latency, "az_o": az_o, "az_r": az_r,
        "mag_o": mag_o, "mag_r": mag_r, "el_r": el_r,
        "o_e": o_e, "c_e": c_e, "r_e": r_e,
        "freqs": freqs, "psd": psd, "thresh": thresh,
        "itds": itds, "ilds": ilds
    }
    return results_dict, None
        # UI visualization
        if uploaded_file is not None:
              results, error = process_audio_data(uploaded_file, n_comp)

        if error:
              st.error(error)
        else:
              st.success(f"Finish processing file: {uploaded_file.name}")

        coll, col2, col3, col4 = st.columns(4)
        coll.metric("SNR", f"{results['snr']:.2f} dB")
        col2.metric("Latency", f"{results['lat']:.2f} ms")
        col3.metric("Azimuth Orig", f"{np.degrees(results['az_o']):.1f}°")
        col4.metric("Azimuth Recon", f"{np.degrees(results['az_r']):.1f}°")

        st.write("### Listen Reconstructed Binaural")
        virtual_file = io.BytesIO()
        sf.write(virtual_file, results['bin'], results['fs'], format='wav')
        st.audio(virtual_file)

        tab1, tab2 = st.tabs(["Waveforms & Errors", "Perceptual & Localization"])

        with tab1:
            fig1, axs1 = plt.subplots(4, 2, figsize=(15, 12))
            labels = ['W', 'X', 'Y', 'Z']
            for i in range(4):
                axs1[i, 0].plot(results['data'][:1000, i], label='Original', alpha=0.5)
                axs1[i, 0].plot(results['recon'][:1000, i], label='Recon', color='orange')
                axs1[i, 0].set_title(f"Channel {labels[i]}")
                diff = results['data'][:, i] - results['recon'][:, i]
                axs1[i, 1].specgram(diff, Fs=results['fs'], cmap='magma')
            plt.tight_layout()
            st.pyplot(fig1)

        with tab2:
            fig2 = plt.figure(figsize=(16, 12))

            # 1. Directional Masking Analysis
            ax1 = fig2.add_subplot(2, 2, 1)
            ax1.semilogy(results['freqs'], results['psd'], label='Signal PSD')
            ax1.semilogy(results['freqs'], results['thresh'], 'r:', label='Masking Threshold')
            ax1.set_title("Directional Masking Analysis")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Power/Freq (dB/Hz)")
            ax1.legend()

            # 2. Total Acoustic Energy Path
            ax2 = fig2.add_subplot(2, 2, 2)
            ax2.plot(results['o_e'][:1500], label='Original', alpha=0.6)
            ax2.plot(results['c_e'][:1500], label='Compressed (Scaled)', alpha=0.8)
            ax2.plot(results['r_e'][:1500], label='Reconstructed', alpha=0.6)
            ax2.set_title("Total Acoustic Energy Path (L2 Norm)")
            ax2.set_xlabel("Time (Samples)")
            ax2.set_ylabel("Energy Magnitude")
            ax2.legend()

            # 3. Binaural Cue Distribution
            ax3 = fig2.add_subplot(2, 2, 3)
            jitter = np.random.normal(0, 1, size=len(results['itds']))
            angles = np.full_like(results['itds'], np.degrees(results['az_r'])) + jitter
            ax3.scatter(angles, results['itds'], color='blue', alpha=0.4, label='ITD (s)')
            ax3.scatter(angles, results['ilds'], color='green', alpha=0.4, label='ILD (dB)')
            ax3.set_title("Binaural Cue Distribution (HRTF-Style)")
            ax3.set_xlabel("Reconstructed Azimuth (Degrees)")
            ax3.set_ylabel("Cue Value")
            ax3.legend()

            # 4. Spatial Localization (Polar Plot)
            ax4 = fig2.add_subplot(2, 2, 4, projection='polar')
            max_mag = max(results['mag_o'], results['mag_r'])
            limit = max_mag * 1.2 if max_mag > 0 else 1.0
            ax4.quiver(results['az_o'], 0, 0, results['mag_o'], angles='xy', scale_units='xy', scale=1, color='blue', label='Original')
            ax4.quiver(results['az_r'], 0, 0, results['mag_r'], angles='xy', scale_units='xy', scale=1, color='red', label='Reconstructed')
            ax4.text(results['az_o'], results['mag_o'], 'Original', color='blue')
            ax4.text(results['az_r'], results['mag_r'], 'Reconstructed', color='red')
            ax4.set_ylim(0, limit)
            ax4.set_title(f"Spatial Localization (Elevation: {np.degrees(results['el_r']):.1f}°)")

            plt.tight_layout()
            st.pyplot(fig2)

else:
    st.info("Please upload an Ambisonic .wav file to begin analysis!")
