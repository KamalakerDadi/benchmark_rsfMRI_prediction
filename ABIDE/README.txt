This folder contains script "run_prediction_on_abide" used for classification
 between autism and healthy subjects.

This script starts with timeseries signals extracted on
ABIDE preprocessed data from Preprocessed Connectome Project Initiative (PCP)
(http://preprocessed-connectomes-project.org/abide/index.html) [1],
866 subjects (autism 402, normal controls 464). This can be downloaded from
https://osf.io/hc4md/download

These timeseries signals are extracted using
pre-defined anatomical atlases such as AAL [2], Harvard Oxford [3] and
pre-computed functional atlases such as Power [4], BASC [5] and MODL (massive
online dictionary learning) [6].

The sotware used for timeseries extraction, a Python library Nilearn
(http://nilearn.github.io/) [7]. This software can handle timeseries
extraction on hard parcellations type atlases typically AAL, Harvard Oxford, BASC
and soft parcellations type atlases like MODL atlases, seeds based atlases
like Power.

These timeseries are used in the preparation of paper [8] discriminating
individuals with Autism from normal controls.

Preprint:
https://hal.inria.fr/hal-01824205

Please review data usage agreements on
(http://preprocessed-connectomes-project.org/) for access to
phenotypic information.

[1] Craddock, C., Benhajali, Y., Chu, C., Chouinard, F., Evans, A.,
    Jakab, A., Khundrakpam, B.S., Lewis, J.D., Li, Q., Milham, M.,
    Yan, C., Bellec, P., 2013. The neuro bureau preprocessing initiative:
    open sharing of preprocessed neuroimaging data and derivatives.
    Frontiers in Neuroinformatics .

[2] Tzourio-Mazoyer, N., et al. 2002. Automated anatomical labeling
    of activations in SPM using a macroscopic anatomical parcellation
    of the MNI MRI single-subject brain. Neuroimage 15, 273.

[3] Desikan, R., et al. 2006. An automated labeling system for
    subdividing the human cerebral cortex on mri scans into gyral
    based regions of interest. Neuroimage 31, 968.

[4] Power, J., et al. 2011. Functional network organization of the human
    brain. Neuron 72, 665-678.

[5] Bellec, P., et al. 2010. Multi-level bootstrap analysis of stable
    clusters in resting-state fMRI. NeuroImage 51, 1126.

[6] Mensch, A., Mairal, J., Thirion, B., Varoquaux, G., 2018. Stochastic
    subsampling for factorizing huge matrices. IEEE Transactions on
    Signal Processing 66, 113-128.

[7] Abraham, A., et al. 2014. Machine learning for neuroimaging with
    scikit-learn. Frontiers in neuroinformatics 8.

[8] Dadi, K. et al. 2018. Benchmarking functional connectome-based predictive
    models for resting-state fMRI. Neuroimage (under review).
