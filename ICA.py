import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
np.random.seed(0)
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def center(X):
    X = np.array(X)

    mean = X.mean(axis=1, keepdims=True)

    return X - mean


def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - \
        g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)

    X = whitening(X)

    components_nr = X.shape[0]

    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):

        w = np.random.rand(components_nr)

        for j in range(iterations):

            w_new = calculate_new_w(w, X)

            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w * w_new).sum()) - 1)

            w = w_new

            if distance < tolerance:
                break

        W[i, :] = w

    S = np.dot(W, X)

    return S


def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    c = 1
    for s in original_sources:
        plt.subplot(4, 1, c)
        c += 1
        plt.plot(s)
    plt.title("real sources")
    for s in S:
        plt.subplot(4, 1, c)
        c += 1
        plt.plot(s)
    plt.title("predicted sources")

    fig.tight_layout()
    plt.show()


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:

            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:

        X += 0.02 * np.random.normal(size=X.shape)

    return X


sampling_rate, source1 = wavfile.read('input1_new.wav')
sampling_rate, source2 = wavfile.read('input2_new.wav')
source1_left_headphone = np.asarray([row[0] for row in source1])
source2_left_headphone = np.asarray([row[0] for row in source2])
source1_right_headphone = np.asarray([row[1] for row in source1])
source2_right_headphone = np.asarray([row[1] for row in source2])

X = np.c_[source1_left_headphone, source2_left_headphone]
A = np.array(([[1, 1], [0.5, 2]]))

X = np.dot(X, A.T)
X = X.T

max_mix1 = np.max(np.abs(np.asarray(X[0])))
mix1 = (np.asarray(X[0]/max_mix1)).astype(np.float)
wavfile.write('Sample_Mixed_Wave_1.wav', int(sampling_rate),
              np.asarray(mix1, dtype=np.float32))
max_mix2 = np.max(np.abs(np.asarray(X[1])))
mix2 = (np.asarray(X[1]/max_mix2)).astype(np.float)
wavfile.write('Sample_Mixed_Wave_2.wav', int(sampling_rate),
              np.asarray(mix2, dtype=np.float32))

S = ica(X, iterations=1000)

wavfile.write('inp1_left_headphone.wav', sampling_rate, source1_left_headphone)
wavfile.write('inp2_left_headphone.wav', sampling_rate, source2_left_headphone)

max_sig_1 = np.max(np.abs(np.asarray(S[0])))
sig_1_32_left = (np.asarray(S[0])/max_sig_1).astype(np.float)
wavfile.write('out1_left_headphone.wav', int(sampling_rate),
              np.asarray(sig_1_32_left, dtype=np.float32))
max_sig_2 = np.max(np.abs(np.asarray(S[1])))
sig_2_32_left = (np.asarray(S[1])/max_sig_2).astype(np.float)
wavfile.write('out2_left_headphone.wav', int(sampling_rate),
              np.asarray(sig_2_32_left, dtype=np.float32))

plot_mixture_sources_predictions(
    X, [source1_left_headphone, source2_left_headphone], [sig_1_32_left, sig_2_32_left])


X = np.c_[source1_right_headphone, source2_right_headphone]
A = np.array(([[1, 1], [0.5, 2]]))

X = np.dot(X, A.T)
X = X.T
S = ica(X, iterations=1000)

wavfile.write('inp1_right_headphone.wav',
              sampling_rate, source1_right_headphone)
wavfile.write('inp2_right_headphone.wav',
              sampling_rate, source2_right_headphone)

max_sig_1 = np.max(np.abs(np.asarray(S[0])))
sig_1_32_right = (np.asarray(S[0])/max_sig_1).astype(np.float)
wavfile.write('out1_right_headphone.wav', int(sampling_rate),
              np.asarray(sig_1_32_right, dtype=np.float32))
max_sig_2 = np.max(np.abs(np.asarray(S[1])))
sig_2_32_right = (np.asarray(S[1])/max_sig_2).astype(np.float)
wavfile.write('out2_right_headphone.wav', int(sampling_rate),
              np.asarray(sig_2_32_right, dtype=np.float32))

plot_mixture_sources_predictions(
    X, [source1_right_headphone, source2_right_headphone], [sig_1_32_right, sig_2_32_right])

wavfile.write('out1.wav', int(sampling_rate), np.asarray(
    [[sig_1_32_left[i], sig_1_32_right[i]] for i in range(0, len(sig_1_32_left))], dtype=np.float32))
wavfile.write('out2.wav', int(sampling_rate), np.asarray(
    [[sig_2_32_left[i], sig_2_32_right[i]] for i in range(0, len(sig_2_32_left))], dtype=np.float32))
