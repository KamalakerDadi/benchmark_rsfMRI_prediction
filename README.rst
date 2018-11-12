Benchmarking models for rsfMRI
==============================

This repository hosts scripts necessary for running predictions on multiple datasets: COBRE, ADNI, ADNIDOD, ACPI, ABIDE, HCP. The idea is to reproduce "Figure 7: Pipelining choices with precomputed regions, across six datasets" from paper [7] which can be seen in the preprint.

Link to the preprint: https://hal.inria.fr/hal-01824205

Data we provide
---------------

Timeseries signals
~~~~~~~~~~~~~~~~~~~

Prior work: we provide the timeseries signals which are extracted on each of 6 datasets using the **same pre-defined/pre-computed atlases** as shown on Figure 7 in [7]. The atlases we used for extraction are : AAL (Automated anatomical labeling) [1], Harvard Oxford [2] and pre-computed functional atlases such as Power [3], BASC (bootstrap analysis of stable clusters) [4] and MODL (massive online dictionary learning) [5].

Note that these are the timeseries used for preparation of our article [7].

We provide the timeseries by uploading them on Open Science Framework (OSF). Please see section Scripts to know how to download or run script which downloads automatically.

For HCP, timeseries data are uploaded into two parts. Due to the limited size of data storage on OSF.
 - Part 1: named as HCP1 which contains timeseries extracted using AAL, Harvard Oxford, BASC. 
 - Part 2: named as HCP2, which contains timeseries extracted using Power and MODL atlases.
 
Phenotypes (partly for some datasets but not all)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this repository, we also uploaded the phenotypic information in csv files for ABIDE and ACPI as they are accessible for downloading.

- ABIDE, a csv file called as "Phenotypic_V1_0b_preprocessed1.csv" is downloaded manually from https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv and uploaded here.

- ACPI, a csv file called "mta_1_phenotypic_data.csv" is downloaded manually from https://s3.amazonaws.com/fcp-indi/data/Projects/ACPI/PhenotypicData/mta_1_phenotypic_data.csv  and uploaded here.

**Please read data usage agreements on datasets websites for access/usage to phenotypic information.**

- For the rest of the datasets, phenotypic information needs to be accessed separately on your own consulting their websites. The website links are provided below.

What software we used
----------------------
- The sotware used for timeseries extraction, a Python library Nilearn (http://nilearn.github.io/) [6]. This software can handle   timeseries extraction on hard parcellations type atlases typically AAL, Harvard Oxford, BASC and soft parcellations type atlases  like MODL atlases, seeds based atlases like Power.

- Installing Nilearn and its dependencies are essential. To install Nilearn and its dependencies (http://nilearn.github.io/introduction.html#installing-nilearn) 

- We also used Pandas (http://pandas.pydata.org/) in the scripts to read csv files.

- For plotting prediction results: matplotlib (https://matplotlib.org/) and seaborn (http://seaborn.pydata.org/ => 0.7.1)

Scripts we provide: What does each script (``run_*``) do when you launch them ?
-------------------------------------------------------------------------------

Each script provided, whose name written as ``run_*.py`` (per dataset) starts by:

1. Downloads timeseries of that particular dataset and saves in **current directory**. The folder name saved in current directory depends on the dataset which you want to run.

2. If timeseries are **downloaded already** in current directory, then each script will load the timeseries directly.

3. After the timeseries data is prepared, each script will load (using pandas) the provided phenotypic information in csv file. This applies in particular to datasets like COBRE, HCP, ADNI, ADNIDOD. If the path to csv file is not provided, then each script will **raise a meaningful error message** of what is missing and what needs to be provided to run script.

4. Scripts named as ``run_prediction_on_abide.py`` and ``run_prediction_on_acpi.py`` are **directly launchable** as phenotypic information is made available for these particular datasets.

Data description and classification task in brief
--------------------------------------------------

- **ABIDE** [8] - the script ``run_prediction_on_abide.py`` can be used for launching predictions on 866 subjects classifying individuals between autism 402 and healthy subjects 464.

  - Website: http://preprocessed-connectomes-project.org/abide/download.html
  
  - **NOTE**: Please note that you are launching 100 cross-validation folds * 7 different atlases * 3 different connectivity measures * 11 different learners on 866 subjects.

- **ACPI** - the script ``run_prediction_on_acpi.py`` can be used for launching predictions discriminating whether individuals have consumed marijuana or not, of 126 subjects (marijuana consumers 62, non-marijuana consumers 64). 

  - Website: http://fcon_1000.projects.nitrc.org/indi/ACPI/html/acpi_mta_1.html
  
- **COBRE** - the script ``run_prediction_on_cobre.py`` is useful for discriminating individuals between schizophrenia 65 and healthy controls 77) comprising of total 142 subjects.

  - Website: http://cobre.mrn.org/
   
  - **NOTE**: Please note that phenotypic information is explicitly provided.

- **ADNI** [9] - the script ``run_prediction_on_adni.py`` can be used for predictions on 136 subjects, classifying individuals between alzheimer's disease 40, mild cognitive impairment 96.

  - Website: http://adni.loni.usc.edu/
   
  - **NOTE**: Please note that phenotypic information is explicitly provided.

- **ADNIDOD** [9] - the script ``run_prediction_on_adnidod.py`` can be used for predictions on 167 subjects, classifying individuals between post traumatic stress disorder (ptsd) 89, normal controls 78.

  - Website: http://adni.loni.usc.edu/
   
  - **NOTE**: Please note that phenotypic information is explicitly provided.
   
- **HCP** [10] - We made two scripts to process two parts of timeseries data (HCP1 and HCP2).

  - The script ``run_prediction_on_hcp1.py`` can be used to process HCP1 to classify individuals belonging to two different groups, low IQ group and high IQ group. Please see paper [7] on details about making two IQ groups comprising of 443 subjects from a total of 900 subjects [10]. 
   
  -  The script ``run_prediction_on_hcp2.py`` can be used to process another part HCP2 for the same classification procedure as described above between high and low IQ groups.
   
  - Website: https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release
   
  - **NOTE**: Please note that phenotypic information is explicitly provided.


Addition information on provided scripts
-----------------------------------------

Visualization script used for generating Figure 7 in preprint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``plot_predictions.py`` can be launched which generates a Figure 7 plot based on predictions directory which has prediction scores for each dataset.

- ``downloader.py`` is provided as a utilities which is used behind fetching timeseries data from OSF for each of 6 datasets. Each function ``fetch_*`` in this module downloads data from OSF into current directory, if path where the data should be downloaded is not provided.
  - Please note that: you don't necessarily have to study this module as data downloading is integrated automatically in each script. 


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

[10] Van Essen, D., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T., Bucholz, R., Chang, A., Chen, L., Corbetta, M., Curtiss, S., Della Penna, S., Feinberg, D., Glasser, M., Harel, N., Heath, A., Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., Oostenveld, R., Petersen, S., Prior, F., Schlaggar, B., Smith, S., Snyder, A., Xu, J., Yacoub, E., 2012. The human connectome project: A data acquisition perspective. NeuroImage 62, 2222-2231.
