#!/usr/bin/env python3

import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


def load_data(subject_id, runs):
    """
    Load EEG data from PhysioNet Motor Imagery dataset.

    Parameters
    ----------
    subject_id : int
        Subject number (1-109)
    runs : list
        Run numbers to load (e.g., [4, 8, 12] for left/right imagery)

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data with standard montage
    """
    raw_fnames = eegbci.load_data(subject_id, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raws) if len(raws) > 1 else raws[0]
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    return raw


def visualize(raw, duration=10.0):
    """
    Visualize raw EEG signals in time domain.

    Plots motor cortex channels (C3, C4, Cz) over time:
    - C3: Left motor cortex (controls right hand)
    - C4: Right motor cortex (controls left hand)
    - Cz: Central reference

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    duration : float
        Time window to display in seconds
    """
    channels = ['C3', 'C4', 'Cz']
    available = [ch for ch in channels if ch in raw.ch_names]

    if not available:
        available = raw.ch_names[:3]

    # Extract signal data and timestamps for selected channels
    data, times = raw.get_data(picks=available, return_times=True)
    # Create boolean mask to keep only first N seconds
    mask = times <= duration
    # Apply mask and convert from Volts to microvolts (×1e6)
    data = data[:, mask] * 1e6
    # Apply same mask to time array
    times = times[mask]

    _, ax = plt.subplots(figsize=(12, 6))
    for i, ch in enumerate(available):
        ax.plot(times, data[i] + i * 50, label=ch, linewidth=0.5)

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Signaux EEG bruts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def apply_filter(raw, l_freq=8.0, h_freq=30.0):
    """
    Apply bandpass filter to keep mu (8-12 Hz) and beta (12-30 Hz) bands.

    These frequency bands are critical for motor imagery:
    - Mu band: Sensorimotor rhythm at rest
    - Beta band: Active motor activity

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency (default: 8 Hz)
    h_freq : float
        High cutoff frequency (default: 30 Hz)

    Returns
    -------
    raw_filtered : mne.io.Raw
        Bandpass filtered EEG data
    """
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='fir', phase='zero', verbose=False)
    return raw_filtered


def main():
    subject_id = 1
    runs = [4, 8, 12]  # Imagery left/right

    raw = load_data(subject_id, runs)
    visualize(raw, duration=30.0)
    raw = apply_filter(raw, l_freq=8.0, h_freq=30.0)
    visualize(raw, duration=30.0)


if __name__ == "__main__":
    main()
