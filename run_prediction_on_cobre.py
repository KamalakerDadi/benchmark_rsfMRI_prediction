"""Script which starts from timeseries extracted on COBRE. The timeseries are
   pre-extracted using several atlases AAL, Harvard Oxford, BASC, Power,
   MODL on these datasets and can be downloaded from
   "https://osf.io/gyrnx/download"

   Diagnostic information we used for prediction task is from file
   "1139_Cobre_Neuropsych_V2_20160607.csv" which should be requested from
   cobre.mrn.org. The column name "Subject Type" contains information whether
   subjects have schizophrenia or normal. We excluded subjects/timeseries
   having bipolar disorder and schizoaffective.

   Prediction task is named as column "Dx_group" (renamed).

   After downloading, each folder should appear with name of the atlas and
   sub-folders, if necessary. For example, using BASC atlas, we have extracted
   timeseries signals with networks and regions. Regions implies while
   applying post-processing method to extract the biggest connected networks
   into separate regions. For MODL, we have extracted timeseries with
   dimensions 64 and 128 components.

   Dimensions of each atlas:
       AAL - 116
       BASC - 122
       Power - 264
       Harvard Oxford (cortical and sub-cortical) - 118
       MODL - 64 and 128

   The timeseries extraction process was done using Nilearn
   (http://nilearn.github.io/).

   Note: To run this script Nilearn is required to be installed.
"""
import os
from os.path import join
import numpy as np
import pandas as pd


def _get_paths(scores, atlas, timeseries_dir):
    """
    """
    timeseries = []
    subject_ids = []
    dx_groups = []
    for index, subj_id in enumerate(scores['Subject_id']):
        this_timeseries = join(timeseries_dir, atlas, subj_id + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            subject_ids.append(subj_id)
            dx_groups.append(scores['Dx_group'][index])
    return timeseries, dx_groups, subject_ids


def get_scores(csv_file):
    csv_file = '1139_Cobre_Neuropsych_V2_20160607.csv'
    df = pd.read_csv(csv_file)
    labels = ['Subject_id', 'Dx_group', 'Visit']
    for i, l in enumerate(labels):
        df = df.rename(columns={'Unnamed: %d' % i: l})
    return df


# Paths
timeseries_dir = '/path/to/timeseries/directory/COBRE'
predictions_dir = '/path/to/save/prediction/results/COBRE'

atlases = ['AAL', 'HarvardOxford', 'BASC/networks', 'BASC/regions',
           'Power', 'MODL/64', 'MODL/128']

dimensions = {'AAL': 116,
              'HarvardOxford': 118,
              'BASC/networks': 122,
              'BASC/regions': 122,
              'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

# prepare dictionary for saving results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'covariance_estimator', 'dimensionality']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

# phenotypes ('Bipolar' and 'Schizoaffective' should be excluded)
pheno_dir = '/path/to/csvfile/COBRE/1139_Cobre_Neuropsych_V2_20160607.csv'
scores = get_scores(csv_file=pheno_dir)

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
    timeseries, diagnosis, ids = _get_paths(scores, atlas, timeseries_dir)

    _, classes = np.unique(diagnosis, return_inverse=True)
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
                results['dataset'].append('COBRE')
                results['dimensionality'].append(dimensions[atlas])
                results['scores'].append(score)
                results['covariance_estimator'].append('LedoitWolf')
    res = pd.DataFrame(results)
    # scores per atlas
    res.to_csv(join(predictions_dir, atlas, 'scores_{0}.csv'.format(atlas)))
all_results = pd.DataFrame(results)
all_results.to_csv('predictions_on_cobre.csv')
