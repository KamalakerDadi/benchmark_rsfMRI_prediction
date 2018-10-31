Benchmarking models for rsfMRI
==============================

This repository contains necessary scripts required for predictions on multiple datasets: COBRE, ADNI, ADNIDOD, ACPI, ABIDE, HCP.
Each dataset has its own script starting with timeseries signals in csvs. For convenient, we uploaded only the timeseries signals on Open Science Framework but not phenotypic or behavioral information. Please review data usage agreements on datasets websites for access to phenotypic information. These timeseries are used in the preparation of paper [7].

These timeseries signals are extracted using pre-defined anatomical atlases such as AAL [1], Harvard Oxford [2] and
pre-computed functional atlases such as Power [3], BASC [4] and MODL (massive online dictionary learning) [5].

The sotware used for timeseries extraction, a Python library Nilearn
(http://nilearn.github.io/) [6]. This software can handle timeseries
extraction on hard parcellations type atlases typically AAL, Harvard Oxford, BASC
and soft parcellations type atlases like MODL atlases, seeds based atlases
like Power.

- ABIDE (http://preprocessed-connectomes-project.org/abide/download.html)

On ABIDE datasets: the script ``run_prediction_on_abide`` can be used. This script starts with timeseries signals extracted on
ABIDE preprocessed data from Preprocessed Connectome Project Initiative (PCP) [8]
(http://preprocessed-connectomes-project.org/abide/index.html), 866 subjects (autism 402, normal controls 464). This can be downloaded from https://osf.io/hc4md/download


- COBRE (http://cobre.mrn.org/)

- ADNI and ADNIDOD (http://adni.loni.usc.edu/)

- ACPI (http://fcon_1000.projects.nitrc.org/indi/ACPI/html/acpi_mta_1.html)


- HCP (https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release)


Preprint available to read:
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dadi, Kamalaker and Rahim, Mehdi and Abraham, Alexandre and Chyzhyk, Darya and Milham, Michael and Thirion, Bertrand and Varoquaux, Gael. **Benchmarking functional connectome-based predictive models for resting-state fMRI.**  2018 (under review) NeuroImage. https://hal.inria.fr/hal-01824205

References
^^^^^^^^^^

[1] Tzourio-Mazoyer, N., et al. 2002. Automated anatomical labeling of activations in SPM using a macroscopic anatomical        parcellation of the MNI MRI single-subject brain. Neuroimage 15, 273.

[2] Desikan, R., et al. 2006. An automated labeling system for subdividing the human cerebral cortex on mri scans into gyral     based regions of interest. Neuroimage 31, 968.

[3] Power, J., et al. 2011. Functional network organization of the human brain. Neuron 72, 665-678.

[4] Bellec, P., et al. 2010. Multi-level bootstrap analysis of stable clusters in resting-state fMRI. NeuroImage 51, 1126.

[5] Mensch, A., Mairal, J., Thirion, B., Varoquaux, G., 2016. Dictionary Learning for Massive Matrix Factorization. International Conference on Machine Learning, 48.

[6] Abraham, A., et al. 2014. Machine learning for neuroimaging with scikit-learn. Frontiers in neuroinformatics 8.

[7] Dadi, K. et al. 2018. Benchmarking functional connectome-based predictive models for resting-state fMRI. Neuroimage (under review).
    
[8] Craddock, C., Benhajali, Y., Chu, C., Chouinard, F., Evans, A., Jakab, A., Khundrakpam, B.S., Lewis, J.D., Li, Q., Milham, M., Yan, C., Bellec, P., 2013. The neuro bureau preprocessing initiative: open sharing of preprocessed neuroimaging data and derivatives. Frontiers in Neuroinformatics.
