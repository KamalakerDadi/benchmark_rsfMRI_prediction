"""New names fixed for plots in the paper
"""
from collections import OrderedDict


def new_names_classifier():
    """ New names assigned to each classifier
    """
    new_names = OrderedDict([('ridge', 'Ridge'),
                             ('svc_l1', r'SVC-$\ell_1$'),
                             ('svc_l2', r'SVC-$\ell_2$'),
                             ('knn', 'K-NN'),
                             ('GaussianNB', 'Gaussian \n Naive Bayes'),
                             ('RandomF', 'Random Forest'),
                             ('logistic_l1', r'Logistic-$\ell_1$'),
                             ('logistic_l2', r'Logistic-$\ell_2$'),
                             ('anova_svcl1', 'ANOVA + \n SVC-$\ell_1$'),
                             ('anova_svcl2', 'ANOVA + \n SVC-$\ell_2$')])
    return new_names


def new_names_atlas():
    """New names assigned to each atlas
    """
    new_names = OrderedDict([('MODL/64', 'MODL dict. learning \n (64 networks)'),
                             ('MODL/128', 'MODL dict. learning \n (128 networks)'),
                             ('AAL', 'AAL \n (116 regions)'),
                             ('BASC/regions', 'BASC \n (122 networks)'),
                             ('BASC/networks', 'BASC \n (122 networks)'),
                             ('HarvardOxford', 'Harvard Oxford \n (118 regions)'),
                             ('Power', 'Power \n (264 regions)')])
    return new_names


def new_names_measure():
    """New names assigned to each measure
    """
    new_names = OrderedDict([('correlation', 'Correlation'),
                             ('partial correlation', 'Partial \n Correlation'),
                             ('tangent', 'Tangent')])
    return new_names

