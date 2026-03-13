"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    #Call load_data('data/sales_records.csv')
    df = load_data('data/sales_records.csv')
    #Assert the result is a pd.DataFrame
    assert isinstance(df, pd.DataFrame)
    #Assert len(df) > 0
    assert len(df) > 0
    #Assert all expected columns are present:
    #        'date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method'
    assert all(col in df.columns for col in ['date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method'])
    pass


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    #Load the data, then call clean_data
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    #Assert cleaned['quantity'].isna().sum() == 0
    assert cleaned['quantity'].isna().sum() == 0
    #Assert cleaned['unit_price'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0
    pass


# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    df = load_data('data/sales_records.csv')
    cleaned = clean_data(df)
    df_with_features = add_features(cleaned)

    assert 'revenue' in df_with_features.columns

    pd.testing.assert_series_equal(
        df_with_features['revenue'],
        df_with_features['quantity'] * df_with_features['unit_price'],
        check_names=False
    )
