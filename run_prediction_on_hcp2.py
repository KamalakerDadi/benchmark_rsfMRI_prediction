"""Script which starts from timeseries extracted on HCP. The zip file
   "HCP2.zip" contains timeseries pre-extracted using several atlases Power,
   MODL (massive online dictionary learning) of dimensions 64, 128.
   HCP2.zip can be downloaded from "https://osf.io/sxafp/download"

   Prediction task we used is to classify individuals from low IQ and
   high IQ indicated by "PMAT24_A_CR" as a behavioral task. Only timeseries
   are arranged according to these two groups but not behavioral data.
   To access behavioral data, users should request from HCP900 release
   website.

   After downloading, each folder should appear with name of the atlas and
   sub-folders, if necessary. For example, using MODL atlas, we have extracted
   timeseries signals with dimensions 64, 128.

   Dimensions of each atlas:
       Power - 264
       MODL - 64, 128 (dimensions)

   The timeseries extraction process was done using Nilearn
   (http://nilearn.github.io/).

   Note: To run this script Nilearn is required to be installed.
"""
import os
import warnings
from os.path import join
import numpy as np
import pandas as pd

from downloader import fetch_hcp2


def _get_paths(subject_ids, hcp_behavioral, atlas, timeseries_dir):
    """
    """
    timeseries = []
    groups = []
    for index, subject_id in enumerate(subject_ids):
        this_hcp_behavioral = hcp_behavioral[hcp_behavioral['Subject'] == subject_id]
        this_timeseries = join(timeseries_dir, atlas,
                               str(subject_id) + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            groups.append(hcp_behavioral['PMAT24_A_CR'].values[0])
    return timeseries, groups

# Path to data directory where timeseries are downloaded. If not
# provided this script will automatically download timeseries in the
# current directory.

timeseries_dir = None

# If provided, then the directory should contain folders of each atlas name
if timeseries_dir is not None:
    if not os.path.exists(timeseries_dir):
        warnings.warn('The timeseries data directory you provided, could '
                      'not be located. Downloading in current directory.',
                      stacklevel=2)
        timeseries_dir = fetch_hcp2(data_dir='./HCP2')
else:
    # Checks if there is such folder in current directory. Otherwise,
    # downloads in current directory
    timeseries_dir = './HCP2'
    if not os.path.exists(timeseries_dir):
        timeseries_dir = fetch_hcp2(data_dir=timeseries_dir)

# Path to data directory where predictions results should be saved.
predictions_dir = None

if predictions_dir is not None:
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
else:
    predictions_dir = './HCP2/predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

atlases = ['Power', 'MODL/64', 'MODL/128']

dimensions = {'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

# prepare dictionary for saving results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'covariance_estimator', 'dimensionality']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

# Prepare phenotypes

# First, path to phenotypes csv file downloaded from HCP900 release should be
# used. In particular, this file "xxx.csv" should contain
# column name "PMAT_24_A_CR" which makes easy to split the IQ individuals to
# lower IQ and upper IQ groups. Steps/Code to create such groups:

# hcp variable below has a key called phenotype is a pandas data frame
# contains fluid intelligence score (integers) to each subject. Given this,

# Split data into lower half and upper half
# quantiles = hcp.phenotype.quantile([0.333, 0.666])
# low_group = hcp.phenotype < quantiles[0.333]
# lower_hcp = hcp[low_group]
# lower_hcp['class_type'] = pd.Series(['lower'] * len(lower_hcp),
#                                     index=lower_hcp.index)

# upper_group = hcp.phenotype > quantiles[0.666]
# upper_hcp = hcp[upper_group]
# upper_hcp['class_type'] = pd.Series(['upper'] * len(upper_hcp),
#                                     index=upper_hcp.index)
# new_hcp = pd.concat([lower_hcp, upper_hcp])
# new_hcp = new_hcp.reset_index()

# Result will be a new dataframe "new_hcp" which has column called 'class_type'
# having name appended as 'lower' or 'upper' for binary classification.

# Based on this a csv_file should be prepared contains one column "Subject"
# to know the subject id and column
# "class_type" specifying whether this "Subject" belongs to lower or upper
# group. For simplification, we provided subject ids in csv file located in
# this current directory named as "HCP_subject_ids.csv". If 'lower' or 'upper'
# can be added based on the code above, it will be easy to run this script.

csv_file = None

if csv_file is None:
        raise ValueError("Path to a csv file is not provided. It should be "
                         "provided to run this script to classify "
                         "individuals whether belongs to low intelligence group "
                         "or high intelligence group. If given, the csv file "
                         "should contain columns 'Subject' for identifying "
                         "subject id. Subject ids are made public provided with"
                         " name 'HCP_subject_ids.csv' in current directory. "
                         "Another expected column is 'class_type' for "
                         "classification denoted by lower or upper.")
# Subject ids who belongs to low and high IQ groups
subject_ids = pd.read_csv('HCP_subject_ids.csv')
hcp_behavioral = pd.read_csv(csv_file)

# Connectomes per measure
from connectome_matrices import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
measures = ['correlation', 'partial correlation', 'tangent']

from my_estimators import sklearn_classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.25,
                            random_state=0)

for atlas in atlases:
    print("Running predictions: with atlas: {0}".format(atlas))
    timeseries, groups = _get_paths(subject_ids, hcp_behavioral,
                                    atlas, timeseries_dir)

    _, classes = np.unique(groups, return_inverse=True)
    iter_for_prediction = cv.split(timeseries, classes)

    for index, (train_index, test_index) in enumerate(iter_for_prediction):
        print("[Cross-validation] Running fold: {0}".format(index))
        for measure in measures:
            print("[Connectivity measure] kind='{0}'".format(measure))
            connections = ConnectivityMeasure(
                cov_estimator=LedoitWolf(assume_centered=True),
                kind=measure)
            conn_coefs = connections.fit_transform(timeseries)

            for est_key in sklearn_classifiers.keys():
                print('Supervised learning: classification {0}'.format(est_key))
                estimator = sklearn_classifiers[est_key]
                score = cross_val_score(estimator, conn_coefs,
                                        classes, scoring='roc_auc',
                                        cv=[(train_index, test_index)])
                results['atlas'].append(atlas)
                results['iter_shuffle_split'].append(index)
                results['measure'].append(measure)
                results['classifier'].append(est_key)
                results['dataset'].append('HCP2')
                results['dimensionality'].append(dimensions[atlas])
                results['scores'].append(score)
                results['covariance_estimator'].append('LedoitWolf')
    res = pd.DataFrame(results)
    # save classification scores per atlas
    this_atlas_dir = join(predictions_dir, atlas)
    if not os.path.exists(this_atlas_dir):
        os.makedirs(this_atlas_dir)
    res.to_csv(join(this_atlas_dir, 'scores.csv'))
all_results = pd.DataFrame(results)
all_results.to_csv('predictions_on_hcp2.csv')
