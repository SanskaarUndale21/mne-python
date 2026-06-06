# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

r"""
.. _tut-mvpa:

===============
Decoding (MVPA)
===============

.. include:: ../../links.inc

In this tutorial you will learn how to:

- Build a decoding pipeline using scikit-learn estimators with MNE
- Apply transformation classes (Scaler, Vectorizer) to prepare EEG/MEG data
- Use spatial filters (CSP) for feature extraction
- Decode brain activity over time using :class:`~mne.decoding.SlidingEstimator`
- Generalize a decoder across time using :class:`~mne.decoding.GeneralizingEstimator`
- Project sensor-space patterns back to source space

Design philosophy
=================
Decoding (a.k.a. MVPA) in MNE largely follows the machine learning API of the
scikit-learn package. Each estimator implements ``fit``, ``transform``,
``fit_transform``, and (optionally) ``inverse_transform`` methods. For more details on
this design, visit scikit-learn_. For additional theoretical insights into the decoding
framework in MNE :footcite:`KingEtAl2018`.

For ease of comprehension, we will denote instantiations of the class using
the same name as the class but in small caps instead of camel cases.

Let's start by loading data for a simple two-class problem:
"""

# %%

# sphinx_gallery_thumbnail_number = 6

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mne
from mne.datasets import sample
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
    get_spatial_filter_from_estimator,
)

data_path = sample.data_path()

subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_filt-0-40_raw.fif"
tmin, tmax = -0.200, 0.500
event_id = {"Auditory/Left": 1, "Visual/Left": 3}  # just use two
raw = mne.io.read_raw_fif(raw_fname)
raw.pick(picks=["grad", "stim", "eog"])

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the example to run faster. The 2 Hz high-pass helps improve CSP.
raw.load_data().filter(2, 20)
events = mne.find_events(raw, "STI 014")

# Set up bad channels (modify to your needs)
raw.info["bads"] += ["MEG 2443"]  # bads + 2 more

# Read epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=("grad", "eog"),
    baseline=(None, 0.0),
    preload=True,
    reject=dict(grad=4000e-13, eog=150e-6),
    decim=3,
    verbose="error",
)
epochs.pick(picks="meg", exclude="bads")  # remove stim and EOG
del raw

X = epochs.get_data(copy=False)  # MEG signals: n_epochs, n_meg_channels, n_times
y = epochs.events[:, 2]  # target: auditory left vs visual left

# %%
# Transformation classes
# ======================
#
# Scaler
# ^^^^^^
# :class:`mne.decoding.Scaler` standardizes the data based on channel scales.
# In the simplest modes ``scalings=None`` or ``scalings=dict(...)``, each data
# channel type (e.g., mag, grad, eeg) is treated separately and scaled by a
# constant. This is the approach used by e.g.,
# :func:`mne.compute_covariance` to standardize channel scales.
#
# If ``scalings='mean'`` or ``scalings='median'``, each channel is scaled
# using empirical measures across all epochs and time points during
# :meth:`~mne.decoding.Scaler.fit`. The key distinction from
# :class:`sklearn.preprocessing.StandardScaler` is that MNE's
# :class:`~mne.decoding.Scaler` scales each *channel* (across time and
# epochs), whereas scikit-learn's version scales each *feature* (e.g., each
# time point) across epochs.
#
# Vectorizer
# ^^^^^^^^^^
# Scikit-learn estimators generally expect 2D data (n_samples × n_features),
# whereas MNE transformers output higher-dimensional arrays
# (e.g. n_samples × n_channels × n_times). :class:`mne.decoding.Vectorizer`
# bridges this gap and must be inserted between MNE and scikit-learn steps:

# Uses all MEG sensors and time points as separate classification
# features, so the resulting filters used are spatio-temporal
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
)

scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None)

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print(f"Spatio-temporal: {100 * score:0.1f}%")

# %%
# Other transformation classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`mne.decoding.PSDEstimator` computes the power spectral density (PSD)
# using the multitaper method, converting a 3D input into 2D.
# :class:`mne.decoding.FilterEstimator` filters the 3D epochs data in place
# within a pipeline. See their API pages for usage details.
#
# Spatial filters
# ===============
#
# Spatial filters provide weights to modify the data along the sensor
# dimension. They are popular in the BCI community because of their simplicity
# and ability to distinguish spatially-separated neural activity.
#
# Common spatial pattern
# ^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`mne.decoding.CSP` analyzes multichannel data from two classes
# :footcite:`Koles1991` (see also
# https://en.wikipedia.org/wiki/Common_spatial_pattern).
# CSP finds spatial filters that maximize variance for one class while
# minimizing it for the other, making it effective for motor imagery and
# other paradigms with distinct spatial distributions.
#
# .. admonition:: Mathematical background
#    :class: dropdown note
#
#    Let :math:`X \in R^{C\times T}` be a segment of data with :math:`C`
#    channels and :math:`T` time points. CSP finds a decomposition that
#    projects the signal in the original sensor space using:
#
#    .. math::       x_{CSP}(t) = W^{T}x(t)
#       :name: csp
#
#    where each column of :math:`W \in R^{C\times C}` is a spatial filter.
#    Let :math:`\Sigma^{+}` and :math:`\Sigma^{-}` be the covariance matrices
#    of the two conditions. CSP finds :math:`W` via simultaneous
#    diagonalization:
#
#    .. math::       W^{T}\Sigma^{+}W = \lambda^{+}
#    .. math::       W^{T}\Sigma^{-}W = \lambda^{-}
#
#    which corresponds to the generalized eigenvalue problem
#    :math:`\Sigma^{+}w = \lambda \Sigma^{-}w`. Filters with large eigenvalues
#    give high variance in one class but low variance in the other, facilitating
#    discrimination.
#
# .. topic:: Examples
#
#     * :ref:`ex-decoding-csp-eeg`
#     * :ref:`ex-decoding-csp-eeg-timefreq`
#
# We can use CSP with these data with:

csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(csp, LinearModel(LogisticRegression(solver="liblinear")))
scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
print(f"CSP: {100 * scores.mean():0.1f}%")

# %%
# Source power comodulation (SPoC)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Source Power Comodulation (:class:`mne.decoding.SPoC`)
# :footcite:`DahneEtAl2014` identifies spatial filters that maximally correlate
# with a continuous target variable. It can be seen as an extension of CSP for
# continuous (rather than discrete) targets, with typical applications in motor
# pattern extraction using EMG power or audio patterns using sound envelope.
#
# .. topic:: Examples
#
#     * :ref:`ex-spoc-cmc`
#
# xDAWN
# ^^^^^
# :class:`mne.preprocessing.Xdawn` improves the signal-to-noise ratio of ERP
# responses :footcite:`RivetEtAl2009`. Originally designed for P300 evoked
# potentials, the MNE-Python implementation generalizes to any ERP type.
#
# .. topic:: Examples
#
#     * :ref:`ex-xdawn-denoising`
#     * :ref:`ex-xdawn-decoding`
#
# Effect-matched spatial filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`mne.decoding.EMS` produces a spatial filter at each time point and a
# corresponding time course :footcite:`SchurgerEtAl2013`. The result gives the
# similarity between the filter at each time point and the data vector at that
# time point.
#
# .. topic:: Examples
#
#     * :ref:`ex-ems-filtering`
#
# Patterns vs. filters
# ^^^^^^^^^^^^^^^^^^^^
#
# When interpreting spatial filter components, it is often more intuitive to
# think in terms of *patterns* (how the signal is composed) rather than
# *filters* (how the signal is projected). For a filter matrix :math:`W`, the
# corresponding spatial patterns are the columns of :math:`(W^{-1})^T`, also
# called the mixing matrix.
#
# .. admonition:: Mathematical background
#    :class: dropdown note
#
#    Rewriting Equation :eq:`csp`:
#
#    .. math::       x(t) = (W^{-1})^{T}x_{CSP}(t)
#       :name: patterns
#
#    The columns of :math:`(W^{-1})^T` are the spatial patterns (mixing
#    matrix). See :ref:`ex-linear-patterns` for a detailed discussion of why
#    patterns are neurophysiologically more interpretable than filters.
#
# These can be plotted for every spatial filter including CSP, XdawnTransformer,
# SSD and SPoC:

# Fit CSP on full data, plot eigenvalues sorted based on mutual information,
# and plot patterns and filters for the three components largest components.
csp.fit(X, y)
spf = get_spatial_filter_from_estimator(csp, info=epochs.info)
spf.plot_scree()
spf.plot_patterns(components=[0, 1, 2])
spf.plot_filters(components=[0, 1, 2], scalings=1e-9)

# %%
# Decoding over time
# ==================
#
# This strategy fits a multivariate predictive model on each time instant and
# evaluates its performance at the same instant on new epochs.
# :class:`mne.decoding.SlidingEstimator` accepts features :math:`X` and
# targets :math:`y`, where :math:`X` has shape
# (n_epochs × n_channels × n_times). An estimator is fit independently on
# every time slice.
#
# This approach tells us *when* one can discriminate experimental conditions,
# and is analogous to SlidingEstimator-based approaches in fMRI.
# When using linear models, this reduces to estimating a discriminative
# spatial filter for each time instant.
#
# Temporal decoding
# ^^^^^^^^^^^^^^^^^
#
# We'll use Logistic Regression for binary classification:

# We will train the classifier on all left visual vs auditory trials on MEG

clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AUC")  # Area Under the Curve
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Sensor space decoding")

# %%
# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel
clf = make_pipeline(
    StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
)
time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
time_decod.fit(X, y)

coef = get_coef(time_decod, "patterns_", inverse_transform=True)
evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
evoked_time_gen.plot_joint(
    times=np.arange(0.0, 0.500, 0.100), title="patterns", **joint_kwargs
)

# %%
# Temporal generalization
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Temporal generalization extends decoding over time by evaluating whether a
# model trained at one time instant can accurately predict *other* time
# instants. This tests whether the neural code at a given moment transfers to
# other moments — analogous to transferring a trained model to a distinct but
# related problem.
#
# :class:`mne.decoding.GeneralizingEstimator` generates predictions from each
# time-trained model for all time instants, producing a training-time ×
# testing-time score matrix. This analysis is described in
# :footcite:`KingEtAl2014` and :footcite:`KingDehaene2014`:

# define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AUC")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Decoding MEG sensors over time")

# %%
# Plot the full (generalization) matrix:

fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    scores,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=epochs.times[[0, -1, 0, -1]],
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("AUC")

# %%
# Projecting sensor-space patterns to source space
# ================================================
# If you use a linear classifier (or regressor) for your data, you can also
# project these to source space. For example, using our ``evoked_time_gen``
# from before:

cov = mne.compute_covariance(epochs, tmax=0.0)
del epochs
fwd = mne.read_forward_solution(meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif")
inv = mne.minimum_norm.make_inverse_operator(evoked_time_gen.info, fwd, cov, loose=0.0)
stc = mne.minimum_norm.apply_inverse(evoked_time_gen, inv, 1.0 / 9.0, "dSPM")
del fwd, inv

# %%
# And this can be visualized using :meth:`stc.plot <mne.SourceEstimate.plot>`:
brain = stc.plot(
    hemi="split", views=("lat", "med"), initial_time=0.1, subjects_dir=subjects_dir
)

# %%
# Source-space decoding
# =====================
#
# Source space decoding is also possible, but because the number of features
# can be much larger than in the sensor space, univariate feature selection
# using ANOVA f-test (or some other metric) can be done to reduce the feature
# dimension. Interpreting decoding results might be easier in source space as
# compared to sensor space.
#
# .. topic:: Examples
#
#     * :ref:`ex-dec-st-source`
#
# Exercise
# ========
#
#  - Explore other datasets from MNE (e.g. Face dataset from SPM to predict
#    Face vs. Scrambled)
#  - Try a different classifier (e.g. :class:`sklearn.svm.SVC`) and compare
#    performance to Logistic Regression
#  - Apply :class:`~mne.decoding.SlidingEstimator` to source-space data and
#    compare the temporal dynamics to sensor-space decoding
#  - Vary the number of CSP components and observe the effect on classification
#    accuracy
#
# References
# ==========
# .. footbibliography::
