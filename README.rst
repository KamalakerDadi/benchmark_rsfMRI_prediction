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

  - On ABIDE datasets: the script ``run_prediction_on_abide.py`` can be used. This script starts with timeseries signals extracted on ABIDE preprocessed data from Preprocessed Connectome Project Initiative (PCP) [8], 866 subjects (autism 402, normal controls 464). This can be downloaded from https://osf.io/hc4md/download


- COBRE (http://cobre.mrn.org/)

  - On COBRE datasets: the script ``run_prediction_on_cobre.py`` is used. This script starts with timeseries signals extracted on COBRE subjects 142 subjects (schizophrenia 65, healthy controls 77). COBRE timeseries can be downloaded from this link   https://osf.io/gyrnx/download

- ADNI and ADNIDOD (http://adni.loni.usc.edu/)

  - On ADNI datasets: the script ``run_prediction_on_adni.py`` is used. This script starts with timeseries signals extracted on ADNI 136 subjects [9] of (alzheimer's disease 40, mild cognitive impairment 96). Download link: https://osf.io/xhrcs/download
  
  - On ADNIDOD datasets: the script ``run_prediction_on_adnidod.py`` is used. This script starts with timeseries signals extracted on ADNIDOD 167 subjects [9] of (ptsd post traumatic stress disorder 89, normal controls 78). Download link: https://osf.io/5aeny/download

- ACPI (http://fcon_1000.projects.nitrc.org/indi/ACPI/html/acpi_mta_1.html)

  - On ACPI datasets: the script ``run_prediction_on_acpi.py`` is used. This script starts with timeseries signals extracted on ACPI preprocessed data. These timeseries are used in the preparation of paper [7] discriminating whether individuals have consumed marijuana or not, of 126 subjects (marijuana consumers 62, non-marijuana consumers 64). Timeseries download link https://osf.io/ab4q6/download


- HCP (https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release)

  - The HCP datasets are split into two parts. HCP1 which contains timeseries extracted using AAL, Harvard Oxford, BASC. The script ``run_prediction_on_hcp1.py`` can be used for this part. This script starts with timeseries signals extracted on two groups of IQ individuals. Please see paper [7] on details about making two IQ groups from 900 subjects. These timeseries are used to classify between low IQ and high IQ, of 443 subjects. Timeseries (HCP1) download link can be found here: https://osf.io/5p7vb/download
  
  - Another part HCP2, which contains timeseries extracted using Power and MODL atlases. The script ``run_prediction_on_hcp2.py`` can be used for this part. This script starts with timeseries signals extracted on two groups of IQ individuals. Please see paper [7] on details about making two IQ groups from 900 subjects. These timeseries are used to classify between low IQ and high IQ, of 443 subjects. Timeseries (HCP2) download link can be found here: https://osf.io/sxafp/download


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

[9] Mueller, S.,  Weiner, M., Thal, L., Petersen, R., Jack, C., Jagust, W., Trojanowski, J.Q., Toga, A.W., Beckett, L., 2005. The alzheimers disease neuroimaging initiative. Neuroimaging Clinics of North America 15, 869.
