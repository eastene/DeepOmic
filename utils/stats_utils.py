#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:34:33 2018

Utility functions for statistical analysis of -omics and clinical data.

@author: evan
"""

from enum import Enum, unique
import numpy as np
import pandas as pd
from operator import itemgetter

"""
vv Scaling Functions
"""

def auto_scale(col: pd.Series) -> pd.Series:
    """
    Apply auto scaling to axis
    :param col: column of dataframe of type np.float_
    :return: copy of col with auto scaling applied
    """
    mean = np.mean(col)
    sd = np.std(col)
    
    def scale(val):
        return (val - mean) / sd
    
    return col.apply(scale)


def range_scale(col: pd.Series) -> pd.Series:
    """
    Apply range scaling to axis
    :param col: column of dataframe of type np.float_
    :return: copy of col with range scaling applied
    """
    mean = np.mean(col)
    xmin = min(col)
    xmax = max(col)
    
    def scale(val):
        return (val - mean) / (xmax - xmin)
    
    return col.apply(scale)


def level_scale(col: pd.Series) -> pd.Series:
    """
    Apply level scaling to axis
    :param col: column of dataframe of type np.float_
    :return: copy of col with level scaling applied
    """
    mean = np.mean(col)
    
    def scale(val):
        return (val - mean) / mean
    
    return col.apply(scale)


@unique
class Scaling(Enum):
    AUTO = auto_scale
    RANGE = range_scale
    LEVEL = level_scale

"""
^^ Scaling Functions
"""


"""
vv Reporting Functions
"""


def rank_attributes_abs(coefs: np.ndarray, attrs: np.ndarray, 
                        ascending: bool=False) -> list:
    """
    Sort absolute values of coefficients (useful for covariance coefficents)
    :param coefs: coefficients on which to sort
    :param attrs: attribute labels corresponding to each coef
    :param ascending: sort in ascending order (default False)
    :return: sorted list of tuples (abs(coef), attr_label) of length (n_attr)
    """
    coef_vals = list(zip(list(np.abs(coefs)), list(attrs)))
    return sorted(coef_vals, key=itemgetter(0), reverse=(not ascending))


def rank_attributes(coefs: np.ndarray, attrs: np.ndarray, 
                        ascending: bool=False) -> list:
    """
    Sort values of coefficients
    :param coefs: coefficients on which to sort
    :param attrs: attribute labels corresponding to each coef
    :param ascending: sort in ascending order (default False)
    :return: sorted list of tuples (coef, attr_label) of length (n_attr)
    """
    coef_vals = list(zip(list(coefs), list(attrs)))
    return sorted(coef_vals, key=itemgetter(0), reverse=(not ascending))


"""
^^ Reporting Functions
"""