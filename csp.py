"""
CSP and workarounds for plotting data with NME-library. This code was
written to provide an easy way to plot the stuff I needed for my
presentation and paper "Preprocessing and Classification of ERD/ERS 
Signals".

- Parts of the dataset mne.datasets.eegbci were used, 
description and discussion of the whole set can be found in:

Schalk, G McFarland, DJ Hinterberger T, Birbaumer N, Wolpaw JR, 
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. 
IEEE TBME 51(6):1034-1043, 2004


- A good introduction to CSP:

Blankertz B, Tomioka R, Lemm S, Kawanabe M, Müller KR, 
Optimizing Spatial Filters for Robust EEG Single-Trial Analysis. 
IEEE Signal Process Mag, 25(1):41-56, 2008


Authors:
    Florian Eichin <eichinflo@aol.com>
"""

import numpy as np
import mne
from scipy.linalg import eigh, inv
import sys


# #################### Helper Functions for MNE ###################
# DISCLAIMER: I didn't really dig much into NME. There might be much
# easier ways of computing the following. My problem was, that I
# wanted to provide the CSP-Algorithm myself for full control and needed
# MNE only for data, plotting and some preprocessing.
# See mne-tools.github.io/dev/auto_examples/decoding/plot_decoding_csp_e
# eg.html for a better example
def make_projection(values, channels, description):
    """
    Make projection from filter values matrix.
    See mne.Projection for details.
    """
    proj_dict = {}
    proj_dict["kind"] = 1
    proj_dict["active"] = True
    proj_dict["desc"] = description
    data_dict = {}
    data_dict["nrow"] = 1
    data_dict["ncol"] = len(channels)
    data_dict["row_names"] = None
    data_dict["col_names"] = channels
    data_dict["data"] = np.array(values)
    proj_dict["data"] = data_dict

    return mne.Projection(proj_dict)


def make_samples(events, data, classes, sample_rate=160):
    """
    Return dict of (class, samples) entries.

    Args:
        events (list of lists) - a list of [start, duration, class]
                                 entries representing the events
        data (mne.io.Raw) - see mne doc for details
        classes (list) - the event-classes that should be considered
        sample_rate (int) - in Hz, defaults to 160 (sr of used dataset)

    Returns:
        dict (classes to lists of np.matrix)
                 - each epoch belonging to one event class is stored
                   as TxCh np.matrix in respective list
    """
    # this dict stores a list of epochs for every class
    cls_to_epoch = dict()
    for cls in classes:
        cls_to_epoch[cls] = []

    cont_signal = data.get_data()
    print(cont_signal.shape)
    # for every event, add according epoch
    for event in events:
        if event[2] in cls_to_epoch.keys():
            # events = [start, duration, class]
            start = int(event[0] * sample_rate)
            end = start + int(event[1] * sample_rate)

            # our algorithm needs matrices
            cls_to_epoch[event[2]].append(
                np.matrix(cont_signal[:, start:end]).T)
    return cls_to_epoch


def get_data(hp=0, lp=40):
    """
    Load and return eeg-dataset 'eegbci'.
    By default filters with bandpass between 3 and 40 Hz.
    (To avoid this, set lp and hp to None)

    Args:
        lp, hp (int) - lowpass and highpass filters

    Returns:
        mne.io.Raw object (see mne doc for details)
    """
    # read EEG-data from mne samples
    path = "/home/flo/mne_data"
    path += "/MNE-eegbci-data/physiobank/database/eegmmidb/S001/"
    f = path + "S001R04.edf"

    # load data
    data = mne.io.read_raw_edf(f, preload=True)
    data = data.drop_channels(['STI 014'])

    # delete existing projections
    data.del_proj()

    # apply lowpass filter
    data.filter(hp, lp)

    return data


# ########################### CSP #############################
def calc_cov(samples):
    """
    Calculates covariance matrix as described in paper.

    Args:
        samples (list of np.matrix) - a list of sample-matrices

    Returns:
        np.matrix - the covariance matrix

    >>> s1 = np.matrix([[1, 0], [0, 1]])
    >>> s2 = np.matrix([[2, 0], [0, 2]])
    >>> calc_cov([s1, s2])
    matrix([[2.5, 0. ],
            [0. , 2.5]])
    """
    no_of_samples = len(samples)
    sigma_c = 1 / no_of_samples * samples[0].T.dot(samples[0])
    for i in range(1, no_of_samples):
        sigma_c += 1 / no_of_samples * samples[i].T.dot(samples[i])
    return sigma_c


def calc_filters(cov1, cov2):
    """
    Calculates the filter matrix W.

    Args:
        covi (np.matrix) - covaiance matrix of class i

    Returns:
        np.matrix - matrix containing filters (sorted by eigenvalues)

    >>> cov1 = np.matrix([[1, 0], [0, 1]])
    >>> cov2 = np.matrix([[1, 0], [0, 1]])
    >>> calc_filters(cov1, cov2)
    matrix([[1., 0.],
            [0., 1.]])
    """
    eigvals, eigvecs = eigh(cov1, cov2, eigvals_only=False)
    return np.matrix(eigvecs)


def apply_csp(data):
    """
    Apply the above functions to get the filter matrix W.
    This basically brings the whole CSP algorithm together.
    """
    # event has form [<start-t>, <stop-t>, <l>]
    events = data.find_edf_events()

    cls_epochs_dict = make_samples(events, data, classes=["T1", "T2"])

    # calc CSP filters
    cov1 = calc_cov(cls_epochs_dict["T1"])
    cov2 = calc_cov(cls_epochs_dict["T2"])

    return calc_filters(cov1, cov2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("=" * 50)
        print("USAGE: python3 csp.py -[MODES] [CHANNEL_NO (Opt)]")
        print()
        print("> MODES:")
        print(">   p : plot EEG-data")
        print(">   t : show topografical map of filters")
        print(">   s : show frequency spectrum")
        print(">   b : use backward model, default")
        print(">   f : use forward model")
        print()
        print("Try 'python3 csp.py -tf' or 'python3 csp.py -s 15'")
        print("=" * 50)
        print()
        sys.exit(1)

    # read mode and ch_no from argv
    modes = set(sys.argv[1][1:])
    ch_no = None
    if len(sys.argv) > 2:
        ch_no = int(sys.argv[2])

    data = get_data()

    W = apply_csp(data)  # backward model
    A = inv(W)  # forward model

    # For some strange reason, I couldnt find proper named layouts, so I
    # had to rename the channels.
    # See physionet.org/pn4/eegmmidb/64_channel_sharbrough-old.png
    # for channel names.
    channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                     'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
                     'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
                     'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7',
                     'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5',
                     'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                     'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7',
                     'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
                     'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4',
                     'PO8', 'O1', 'Oz', 'O2', 'Iz']
    layout = mne.channels.read_layout('EEG1005')

    if 'f' in modes:
        # user wants forward model
        filters = A
    else:
        # default is backward model
        filters = W

    # use first and last ten projections
    projs = []
    for j in range(0, 10):
        proj0 = make_projection(filters[j, :], channel_names,
                                "CSP_forward(%d)" % j)
        proj1 = make_projection(filters[-1 - j, :], channel_names,
                                "CSP_forward(%d)" % -j)
        projs.append(proj0)
        projs.append(proj1)

    data.add_proj(projs)
    # choose channel for spectral analysis
    if ch_no:
        interesting = [ch_no]
    else:
        interesting = [15]

    # plot the stuff that was asked for in modes
    if 't' in modes:
        data.plot_projs_topomap(layout=layout)
    if 'p' in modes:
        data.plot(block=True)
    if 's' in modes:
        data.plot_psd(proj=False, picks=interesting, n_overlap=100)
    data.apply_proj()
    if 'p' in modes:
        data.plot(block=True)
    if 's' in modes:
        data.plot_psd(picks=interesting, n_overlap=100)
