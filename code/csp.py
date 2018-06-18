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
import matplotlib.pyplot as plt
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


def get_projections(filters, a, b, descr, channel_names, d=1):
    """
    Return projections defined in rows a to b of filters.

    Args:
        a, b (int) - index boundaries of the slice of filters

        filters (np.matrix) - 2-dim filter matrix
    """
    projs = []
    if d == -1:
        # go backwards
        a, b = b, a
    for j in range(a, b, d):
        proj = make_projection(filters[j, :], channel_names,
                               descr + str(abs(j) + 1))
        projs.append(proj)
    return projs


def make_variance_plt():
    """
    A function for making the example variance plot for explaining CSP.
    Nothing to be seen here.
    """
    # covariance matrices for the toy data
    cov1 = [[1.5, 0.6], [0.6, 0.4]]
    cov2 = [[0.4, 0.6], [0.6, 1.5]]

    # get mean-free distributions
    X1 = np.random.multivariate_normal([0, 0], cov1, 400)
    X2 = np.random.multivariate_normal([0, 0], cov2, 400)

    plt.scatter(X1[:, 0], X1[:, 1], c="green", marker=".", alpha=0.7)
    plt.scatter(X2[:, 0], X2[:, 1], c="blue", marker=".", alpha=0.7)
    plt.show()

    # apply CSP-filters to data
    W = calc_filters(np.matrix(cov1), np.matrix(cov2))

    X1 = X1.dot(W).getA()
    X2 = X2.dot(W).getA()

    plt.scatter(X1[:, 0], X1[:, 1], c="green", marker=".", alpha=0.7)
    plt.scatter(X2[:, 0], X2[:, 1], c="blue", marker=".", alpha=0.7)
    plt.show()


def get_data_for_class(data, cls):
    """
    Returns raw object with only concatenated epochs of type cls.
    """
    events = data.find_edf_events()

    cls_events = [event for event in events if event[2] == cls]
    print(cls_events)

    new_data = data.copy().crop(
        tmin=cls_events[0][0],
        tmax=cls_events[0][0] + cls_events[0][1])

    for i in range(1, len(cls_events)):
        next_epoch = data.copy().crop(
            tmin=cls_events[i][0],
            tmax=cls_events[i][0] + cls_events[i][1])
        new_data.append(next_epoch)

    return new_data


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
        print("USAGE: python3 csp.py -[MODES] [CHANNEL_NOS (Opt)]")
        print()
        print("> MODES:")
        print(">   p : plot EEG-data")
        print(">   t : show topografical map of filters")
        print(">   s : show frequency spectrum")
        print(">   b : use backward model, default")
        print(">   f : use forward model")
        print()
        print("Try 'python3 csp.py -tf' or 'python3 csp.py -s 15 17'")
        print("=" * 50)
        print()
        sys.exit(1)

    # read mode and ch_no from argv
    modes = set(sys.argv[1][1:])
    ch_no = None
    if len(sys.argv) > 2:
        ch_no = int(sys.argv[2])

    data = get_data(hp=5, lp=30)

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

    # use first ten projections
    projs = get_projections(filters, 0, 10, "w_",
                            channel_names=channel_names)

    # projs += get_projections(filters, 60, 63, "w_",
    #                        channel_names=channel_names, d=-1)

    data.add_proj(projs)

    # make continuous data of event related parts
    t1_data = get_data_for_class(data, "T1")
    t2_data = get_data_for_class(data, "T2")
    t1_data.add_proj(projs)
    t2_data.add_proj(projs)
    plots = []

    # choose channel for spectral analysis
    if ch_no:
        interesting = [ch_no]
    else:
        interesting = [15, 8, 7, 14]
        baseline = [23]

    # plot the stuff that was asked for in modes
    if 't' in modes:
        data.plot_projs_topomap(layout=layout)
    if 'p' in modes:
        data.plot(block=True)
    if 's' in modes:
        no = 128
        nf = 256
        t1_data.plot_psd(proj=False,
                         picks=interesting,
                         color="blue",
                         n_overlap=no,
                         n_fft=nf,
                         spatial_colors=False,
                         estimate="amplitude")
        t2_data.plot_psd(proj=False,
                         picks=interesting,
                         n_overlap=no,
                         n_fft=nf,
                         spatial_colors=False,
                         color="red",
                         estimate="amplitude")

    # apply projections to data
    data.apply_proj()
    t1_data.apply_proj()
    t2_data.apply_proj()

    if 'p' in modes:
        data.plot(block=True)
    if 's' in modes:
        t1_data.plot_psd(picks=interesting,
                         n_overlap=no,
                         n_fft=nf,
                         spatial_colors=False,
                         color="blue",
                         estimate="amplitude")
        t2_data.plot_psd(picks=interesting,
                         n_overlap=no,
                         n_fft=nf,
                         spatial_colors=False,
                         color="red",
                         estimate="amplitude")
    if 'c' in modes:
        make_variance_plt()
