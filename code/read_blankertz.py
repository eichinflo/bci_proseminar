"""
Preprocess data using CSP.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, inv
import sys
import time
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from scipy import signal


### METHODS for reading EEG-data ###

def parse_data(cont_signal, markers, info):
    """
    Parse data from file cont_signal and cut into chunks, where part-
    icipants imagined movements. Timepoints are determined by markers
    file.

    For further explanations see
    http://bbci.de/competition/iv/desc_1.html

    Args:
        cont_signal (String) - name of a file containing signal data
                               where each line contains one timessample
                               with values of all channels seperated by
                               <TAB>
        markers (String) - name of a file containing markers, that
                           indicate positions of classes
                           Each line must be formatted like
                           marker-index<TAB>class-label
        info (String) - file containing meta-data

    Returns:
        tuple of lists of matrices - where entry i of tuple contains
                                     list of samples for class i

    >>> parse_data("test_cnt.txt", "test_mrk.txt", "test_nfo.txt")
    ([matrix([[ 1,  1,  3],
            [ 4,  4,  2],
            [ 5,  2,  7],
            [-5, -3, -1]])], [matrix([[ 5,  2,  7],
            [-5, -3, -1],
            [ 6, -1,  3],
            [ 5,  2, -1]])])

    """
    # parse markers for class samples
    mrks, y = parse_markers(markers)
    
    # read cotinuous signal from file
    cont = parse_cont_signal(cont_signal)
    
    # read sample rate
    sample_rate = 1
    with open(info) as f_i:
        for line in f_i:
            if line.startswith('fs'):
                sample_rate = int(line.strip('\n').split(':')[1])

    # put sample-matrices in respective lists
    samples_1 = []
    samples_m1 = []
    for i, marker in enumerate(mrks):
        if y[i] == 1:
            # stimulae are shown for 4 seconds
            samples_1.append(
                np.matrix(cont[marker: marker + 4 * sample_rate]))
        else:
            samples_m1.append(
                np.matrix(cont[marker: marker + 4 * sample_rate]))

    return samples_1, samples_m1


def parse_markers(f):
    """
    Parse markers from mrk file f. Each marker determines a time point
    and an according class-label of the movement that was imagined.

    Args:
        f (String) - an mrk file

    Returns:
        tuple of lists of ints - one list for the markers, one for 
                                 the labels
    """
    mrks = list()
    y = list()
    with open(f) as f_m:
        for line in f_m:
            mrk, cls = line.strip('\n').split('\t')
            mrks.append(int(float(mrk)))
            y.append(int(float(cls)))
    return mrks, y


def parse_cont_signal(f):
    """
    Parse continuous signal from raw cnt file.

    Returns:
        numpy.matrix - of shape NxC where N is the number of 
                       time-samples and C the number of channels
    """
    cont = []
    with open(f) as f_cs:
        for line in f_cs:
            smpl = []
            for i in line.strip('\n').split('\t'):
                smpl.append(int(i))
            cont.append(smpl)
    return cont


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


def plot_sample(sample):
    """
    Plot a sample matrix using matplotlib.pyplot

    This code was copied partially from
    https://matplotlib.org/gallery/specialty_plots/mri_with_eeg.html
    """
    fig = plt.figure("MRI_with_EEG")
    ticklocs = []
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_xlim(0, 4)
    ax2.set_xticks(np.arange(4))
    numRows = sample.shape[1]
    numSamples = sample.shape[0]
    t = 10.0 * np.arange(numSamples) / numSamples
    dmin = sample.min()
    dmax = sample.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows - 1) * dr + dmax
    ax2.set_ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack(
            (t[:, np.newaxis], sample[:, i, np.newaxis])))
        ticklocs.append(i * dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs
    
    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax2.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    # ax2.set_yticks(ticklocs)
    # ax2.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])

    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

def plot_spectogram(channel):
    """
    Plot a spectocram for sample.
    """
    channel = np.array(channel[0, :])[0]
    # f, t, Sxx = signal.spectrogram(channel, fs=100)
    # print(f, t, Sxx)
    # plt.pcolormesh(t, f, Sxx)
    plt.specgram(channel, Fs=100, scale='dB', NFFT=100,
                 noverlap=5, mode='psd', detrend='mean')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.ylim([0, 100])
    plt.show()
    plt.magnitude_spectrum(channel, Fs=100, scale='dB')
    plt.xlim([0, 30])
    plt.show()

if __name__ == "__main__":
    t1 = time.time()
    if len(sys.argv) != 4:
        print("Usage: python3 csp.py [cnt-file] [mrk-file] [nfo-file]")
        sys.exit()

    samples_1, samples_m1 = parse_data(sys.argv[1],
                                       sys.argv[2],
                                       sys.argv[3])
    t2 = time.time()
    print("Samples parsed... (%ds)" % (t2 - t1))
    cov_p = calc_cov(samples_1)
    cov_m = calc_cov(samples_m1)
    W = calc_filters(cov_p, cov_m)
    t3 = time.time()
    print("Filters calculated! Took %ds." % (t3 - t1))

    cont = np.matrix(parse_cont_signal(sys.argv[1]))
    plot_sample(samples_1[13][:, 0:4])
    plot_sample(samples_1[13].dot(W)[:, 0:4])

    plot_spectogram(cont[100:4000,0:4].T)
    plot_spectogram(cont.dot(W)[100:4000, 0:4].T)
    # plot_W()
# TODO: plot on head according
# TODO: make frequency spectra before and after
# TODO: plot samples before and after, also cleaned
