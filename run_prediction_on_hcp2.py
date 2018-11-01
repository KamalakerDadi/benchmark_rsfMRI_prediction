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
from os.path import join
import numpy as np
import pandas as pd


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


# Paths
timeseries_dir = '/path/to/timeseries/directory/HCP2'
predictions_dir = '/path/to/save/prediction/results/HCP2'

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

# Subject ids of low and high IQ groups
subject_ids = pd.read_csv('HCP_subject_ids.csv')
hcp_behavioral = pd.read_csv('/path/to/phenotypes/HCP2/')

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
    # scores per atlas
    res.to_csv(join(predictions_dir, atlas, 'scores_{0}.csv'.format(atlas)))
all_results = pd.DataFrame(results)
all_results.to_csv('predictions_on_hcp2.csv')
