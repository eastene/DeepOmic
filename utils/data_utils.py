#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:41:13 2018

Utility functions for managing and manipulating -omics and clinical data.

@author: evan
"""

import numpy as np
import pandas as pd

from utils.stats_utils import Scaling
from typing import List

"""
vv Dataframe Manipulation Functions
"""


def split_omics(df: pd.DataFrame, 
                types: List[str]=["clinical", "soma", "metab"],
                split_cols: bool=True) -> List:
    """
    Split joined multi-omics data sets into seperate, single omic components
    Useful when reading a file containing multiple omics
    :param df: dataframe of multiple joined omic data sets
    :param types: list of types of omic (or clinical) data contained in df,
        types should be in order of when they appear in the data set, can be 
        any of the following types:
            "clinical" -> clinical data with 906 attributes
            "soma" -> SOMAscan proteomics data with 1317 attributes
            "metab" -> metabolite data with 
            "emory" -> emory metabolomics data with 2682 attributes
    :param split_cols: split along columns (if False, split along rows)
    :return: list of omic dataset(s) split along the joined direction
    """
    pos = 0
    # dict of type_name -> num of attributes
    valid_types = {"clinical": 906,
                   "soma": 1317,
                   "metab": 1005,
                   "emory": 2862}
    data_list = []
    for data in types:
        if data in valid_types:
            start = pos
            end = pos + valid_types[data]
            data_list.append(df.iloc[:, start:end] if split_cols 
                              else df.iloc[start:end, :])
            pos = pos + valid_types[data]
            
    return data_list


def intersect_with(df: pd.DataFrame, 
                   other: pd.DataFrame, on_left: str,
                   on_right: str=None) -> pd.DataFrame:
    """
    Intersect dataframe with another, smaller dataframe on specified column
    :param df: dataframe with larger number of rows
    :param other: dataframe with smaller number of rows
    :param on_left: column present in df on which to intersect
    :param on_right: column present in other on which to intersect, if values
        are different for on_left and on_right, they must contain overlapping
        values from the same omic/clinical attribute in df and other to 
        intersect correctly
        NOTE: if None, intersects using on_left as the key in both dataframes  
    :return: rows of df that intersect other on the specified column(s)
    """
    identifiers = set(df[on_left]).intersection(
            set(other[on_left if on_right is None else on_right])
    )
    return df[df[on_left].isin(identifiers)]


"""
^^ Dataframe Manipulation Functions
"""


"""
vv Data Transformation Functions
"""


def object2catcodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns of dtype object to integer category codes of dtype int_
    :param df: dataframe containing one or more columns of dtype object
    :return: copy of df containing no columns of dtype object
    """
    copy_frame = df.copy()
    obj_cols = copy_frame.select_dtypes(np.object_).columns.values
    for col in obj_cols:
        copy_frame[col] = copy_frame[col].astype('category').cat.codes
    
    return copy_frame


def standardize_continuous(df: pd.DataFrame, 
                           method=Scaling.AUTO) -> pd.DataFrame:
    """
    Apply scaling to each column of dtype float_
    :param df: dataframe containing any number of columns of dtype float_ 
    :param method: scaling method to use, one of: AUTO, RANGE, or LEVEL
    :return: copy of df with scaling applied to applicable columns
    """
    copy_frame = df.copy()
    continuous_cols = copy_frame.select_dtypes(np.float_).columns.values
    continuous_only = copy_frame[continuous_cols]
    copy_frame[continuous_cols] = continuous_only.apply(method, axis=1)
    
    return copy_frame


"""
^^ Data Transformation Functions
"""
                