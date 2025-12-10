#!/usr/bin/env python3

import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


def load_data(subject_id, runs):
    raw_fnames = eegbci.load_data(subject_id, runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raws) if len(raws) > 1 else raws[0]
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    return raw


def visualize(raw, duration=10.0):
    channels = ['C3', 'C4', 'Cz']
    available = [ch for ch in channels if ch in raw.ch_names]

    if not available:
        available = raw.ch_names[:3]

    data, times = raw.get_data(picks=available, return_times=True)
    mask = times <= duration
    data = data[:, mask] * 1e6
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
