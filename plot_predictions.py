
import pandas as pd
import os
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, cm
import seaborn as sns
from collections import OrderedDict
import numpy as np

from my_palette import (color_palette, atlas_palette,
                        datasets_palette)

#####
# seaborn version == '0.7.1'


def _pandas_data_frame(path):
    """Load path to pandas data frame

    Parameters
    ----------
    path : str
        Path to csv file

    Returns
    -------
    data : pd.DataFrame
        Pandas data frame per path.
    """
    data = pd.read_csv(path)
    scores = data['scores'].str.strip('[ ]')
    del data['scores']
    data = data.join(scores)
    data.scores = data.scores.astype(float)

    return data


def _get_markers():
    marker_dict = {'COBRE': 'o',
                   'ADNI': 's',
                   'ADNIDOD': '>',
                   'ACPI': '^',
                   'ABIDE': 'v',
                   'HCP': '<'}
    return marker_dict


def _add_axvline(ax):

    # draw a default vline at x=0 that spans the yrange
    ax.axvline(x=0, ymin=0.715, linewidth=4, zorder=0, color='0.6')
    ax.axvline(x=0, ymax=0.665, ymin=0.526, linewidth=4, zorder=0, color='0.6')
    ax.axvline(x=0, ymax=0.473, linewidth=4, zorder=0, color='0.6')
    return ax


def _add_yticklabels(ax):

    ax_yticklabels = []
    for y_label in ax.get_yticklabels():
        if y_label.get_text() not in ['dummy1', 'dummy2']:
            ax_yticklabels.append(y_label.get_text())
        else:
            ax_yticklabels.append("")
    ax.set_yticklabels(ax_yticklabels)
    return ax


def _add_xticklabels(ax):

    # make the positive labels with "+"
    ax_xticklabels = []
    for x in ax.get_xticks():
        if x > 0:
            ax_xticklabels.append('+' + str(np.round(x, decimals=2)))
        else:
            ax_xticklabels.append(str(np.round(x, decimals=2)))
    ax.set_xticklabels(ax_xticklabels)
    return ax


def _add_bgcolors(df, ax):

    # background
    for a, method in enumerate(df['method'].unique()):
        if a % 2:
            ax.axhspan(a - .5, a + .5, color='0.9', zorder=-1)
        if a <= 5:
            facecolor = cm.Set2(7)
        elif a > 6 and a < 9:
            facecolor = cm.Set2(3)
        elif a > 10:
            facecolor = cm.Set2(4)
        if method not in ['dummy1', 'dummy2']:
            ax.axhspan(a - .5, a + .5, facecolor=facecolor, alpha=0.15)
    return ax


def _scatter_plot(df, ax):
    marker_dict = _get_markers()

    methods = df.method.unique()

    n_datasets = len(df['dataset'].unique())

    width_of_boxplot = 0.9
    offset = width_of_boxplot / n_datasets
    n_offsets = [-offset * 2.5, -offset * 1.5, -offset * 0.9, 0,
                 offset * 1.2, offset * 2.2]

    for i, (off, dataset) in enumerate(zip(n_offsets,
                                           ['COBRE', 'ADNI', 'ADNIDOD',
                                            'ACPI', 'ABIDE', 'HCP'])):
        # Plot dataset by dataset
        data = df[(df['dataset'] == dataset)]
        for ii, method in enumerate(methods):
            this_method = data[data['method'] == method]
            if not this_method.empty or \
                    this_method.method.unique() not in ['dummy1', 'dummy2']:
                rank = len(this_method) * [ii]
                y_ = np.add(rank, len(this_method) * [off])
                ax.scatter(x=this_method[x], y=y_, data=this_method,
                           color=datasets_palette[dataset],
                           marker=marker_dict[dataset], s=28)
    return ax


def _add_legend(ax):

    marker_dict = _get_markers()

    cobre_marker1 = plt.Line2D([], [], color=datasets_palette['COBRE'],
                               marker=marker_dict['COBRE'], linestyle='')
    adni_marker2 = plt.Line2D([], [], color=datasets_palette['ADNI'],
                              marker=marker_dict['ADNI'], linestyle='')
    adnidod_marker3 = plt.Line2D([], [], color=datasets_palette['ADNIDOD'],
                                 marker=marker_dict['ADNIDOD'], linestyle='')
    acpi_marker4 = plt.Line2D([], [], color=datasets_palette['ACPI'],
                              marker=marker_dict['ACPI'], linestyle='')
    abide_marker5 = plt.Line2D([], [], color=datasets_palette['ABIDE'],
                               marker=marker_dict['ABIDE'], linestyle='')
    hcp_marker6 = plt.Line2D([], [], color=datasets_palette['HCP'],
                             marker=marker_dict['HCP'], linestyle='')
    ax.legend([cobre_marker1, adni_marker2, adnidod_marker3, acpi_marker4,
               abide_marker5, hcp_marker6],
              ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE', 'HCP'],
              loc='lower left', handletextpad=-0.5, borderaxespad=0,
              fontsize=14, frameon=True, scatterpoints=1,
              markerscale=0.9, borderpad=None,
              ncol=1, columnspacing=-0.2)
    return ax


def boxplot(df, x=None, y=None, hue=None, axx=None):

    rc('xtick', labelsize=14)
    rc('ytick', labelsize=14)

    sns.boxplot(data=df, x=x, y=y, fliersize=0, linewidth=2,
                boxprops={'facecolor': 'lightcyan', 'edgecolor': '.0'},
                width=0.9, ax=axx)
    axx = _scatter_plot(df, axx)

    axx = _add_axvline(axx)

    axx.set_ylabel('')

    axx = _add_yticklabels(axx)

    axx = _add_xticklabels(axx)

    axx = _add_bgcolors(df, axx)

    axx = _add_legend(axx)

    plt.tight_layout(rect=[0.1, .01, 1, 0.98], pad=0.1, w_pad=1)
    axx.set_xlabel('Relative prediction scores (AUC)',
                   fontsize=14, fontweight='normal')
    axx.text(0.15, 3.55, 'Regions-definition \n pre-computed atlases',
             fontsize=14, rotation='vertical', va='bottom')
    axx.text(0.15, 7.2, 'Connectivity', fontsize=14,
             rotation='vertical')
    axx.text(0.15, 12.8, 'Classifiers', fontsize=14, rotation='vertical')
    return

###############################################################################
# Gather data


def _get_data():
    """
    """
    data = []
    dataset_names = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE',
                     'HCP']
    for dataset in dataset_names:
        each_atlas_path = join('predictions', dataset, 'scores.csv')
        this_data = _pandas_data_frame(each_atlas_path)
        data.append(this_data)
    data = pd.concat(data)
    return data

data = _get_data()
data = data.drop('Unnamed: 0', axis=1)
##############################################################################
# Prepare data for plotting: calculate mean


def demean(group):
    return group - group.mean()

# Take the average over iter_shuffle_split
df = data.groupby(['classifier', 'measure', 'atlas', 'dataset']).mean()
df = df.reset_index()
df.pop('iter_shuffle_split')
demeaned_scores_atlas = df.groupby(['classifier', 'measure',
                                    'dataset'])['scores'].transform(demean)
demeaned_scores_measure = df.groupby(['atlas', 'classifier',
                                      'dataset'])['scores'].transform(demean)
demeaned_scores_classifier = df.groupby(['atlas', 'measure',
                                         'dataset'])['scores'].transform(demean)
df['demeaned_scores_atlas'] = demeaned_scores_atlas
df['demeaned_scores_measure'] = demeaned_scores_measure
df['demeaned_scores_classifier'] = demeaned_scores_classifier

from aliases import new_names_atlas, new_names_measure, new_names_classifier
df = df.replace(to_replace={'atlas': new_names_atlas(),
                            'measure': new_names_measure(),
                            'classifier': new_names_classifier()})
# change the name of the dataset to upper
df['dataset'] = df['dataset'].str.upper()
df = df[df['classifier'] != 'lasso']
###############################################################################
# Combine into one dataframe

new_df_c = df[['classifier', 'dataset', 'dimensionality',
               'demeaned_scores_classifier']]
new_df_c = new_df_c.rename(index=str,
                           columns={'classifier': 'method',
                                    'demeaned_scores_classifier': 'demeaned_scores'})

new_df_m = df[['measure', 'dataset', 'dimensionality', 'demeaned_scores_measure']]
new_df_m = new_df_m.rename(index=str,
                           columns={'measure': 'method',
                                    'demeaned_scores_measure': 'demeaned_scores'})
dummy2 = pd.DataFrame({"method": ["dummy2", 'dummy2', 'dummy2',
                                  'dummy2', 'dummy2', 'dummy2'],
                       "dataset": ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI',
                                   'ABIDE', 'HCP']})

new_df_m = pd.concat([new_df_m, dummy2])

new_df_a = df[['atlas', 'dataset', 'dimensionality', 'demeaned_scores_atlas']]
new_df_a = new_df_a.rename(index=str,
                           columns={'atlas': 'method',
                                    'demeaned_scores_atlas': 'demeaned_scores'})
dummy1 = pd.DataFrame({"method": ["dummy1", 'dummy1', 'dummy1',
                                  'dummy1', 'dummy1', 'dummy1'],
                       "dataset": ['COBRE', 'ADNI', 'ADNIDOD',
                                   'ACPI', 'ABIDE', 'HCP']})
new_df_a = pd.concat([new_df_a, dummy1])
df = pd.concat([new_df_a, new_df_m, new_df_c])
dic = {'AAL \n (116 regions)': 1,
       'Harvard Oxford \n (118 regions)': 2,
       'BASC \n (122 networks)': 4,
       'Power \n (264 regions)': 3,
       'MODL dict. learning \n (64 networks)': 5,
       'MODL dict. learning \n (128 networks)': 6,
       'dummy1': 7,
       'Partial \n Correlation': 8,
       'Correlation': 9,
       'Tangent': 10,
       'dummy2': 11,
       'K-NN': 12,
       'Random Forest': 13,
       'Gaussian \n Naive Bayes': 14,
       'SVC-$\\ell_1$': 15,
       'ANOVA + \n SVC-$\\ell_1$': 16,
       'Logistic-$\\ell_1$': 17,
       'Ridge': 18,
       'SVC-$\\ell_2$': 19,
       'Logistic-$\\ell_2$': 21,
       'ANOVA + \n SVC-$\\ell_2$': 20}
df['rank'] = df['method'].map(dic)
df.sort_values(by=['rank'], inplace=True)
###############################################################################
# Plotting goes here
hue = 'dataset'
x = 'demeaned_scores'
y = 'method'

fig, axes = plt.subplots(figsize=(5.5, 11.4))
sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.8'})
sns.set_palette('dark')
boxplot(df, x=x, y=y, hue=hue, axx=axes)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()
