"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Complete each function below. Remove the TODO: comments and pass statements
as you implement each function. Do not change the function signatures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load sales records from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw sales records DataFrame.
    """
    #Load the CSV using pd.read_csv(filepath)
    df = pd.read_csv(filepath)
    # Print a progress message: f"Loaded {len(df)}
    #  records from {filepath}"
    print(f"Loaded {len(df)} records from {filepath}")
    #Return the DataFrame
    return df


def clean_data(df):
    """Handle missing values and fix data types.

    - Fill missing 'quantity' values with the column median.
    - Fill missing 'unit_price' values with the column median.
    - Parse the 'date' column to datetime (use errors='coerce' to handle malformatted dates).
    - Print a progress message showing the record count after cleaning.

    Args:
        df (pd.DataFrame): Raw DataFrame from load_data().

    Returns:
        pd.DataFrame: Cleaned DataFrame (do not modify the input in place).
    """
    """Handle missing values and fix data types."""
    
    df = df.copy()

    # Fill missing values with median
    df['quantity'] = df['quantity'].fillna(df['quantity'].median())
    df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())

    # Parse date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    print(f"Cleaned data: {len(df)} records after handling missing values and parsing dates.")
    return df

def add_features(df):
    """Compute derived columns.

    - Add 'revenue' column: quantity * unit_price.
    - Add 'day_of_week' column: day name from the date column.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from clean_data().

    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    df = df.copy()

    revenue = df['quantity'] * df['unit_price']
    revenue.name = None
    df['revenue'] = revenue

    df['day_of_week'] = df['date'].dt.day_name()

    return df



def generate_summary(df):
    """Compute summary statistics.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().

    Returns:
        dict: Summary with keys:
            - 'total_revenue': total revenue (sum)
            - 'avg_order_value': average order value (mean)
            - 'top_category': product category with highest total revenue
            - 'record_count': number of records in df
    """
    #Compute top category: df.groupby('product_category')['revenue'].sum().idxmax()
    top_category = df.groupby('product_category')['revenue'].sum().idxmax()
    #Return a dict with the four keys above
    return {
        'total_revenue': df['revenue'].sum(),
        'avg_order_value': df['revenue'].mean(),
        'top_category': top_category,
        'record_count': len(df)
    }
  


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create and save 3 charts as PNG files.

    Charts to create:
    1. Bar chart: total revenue by product category
    2. Line chart: daily revenue trend (aggregate revenue by date)
    3. Horizontal bar chart: average order value by payment method

    Save each chart as a PNG using fig.savefig().
    Do NOT use plt.show() — it blocks execution in pipeline scripts.
    Close each figure with plt.close(fig) after saving.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().
        output_dir (str): Directory to save PNG files (create if needed).
    """
    # Create the output directory: os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1 — Bar chart: total revenue by product category
    revenue_by_category = df.groupby('product_category')['revenue'].sum()
    categories = revenue_by_category.index
    values = revenue_by_category.values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, values)
    ax.set_title('Total Revenue by Product Category')
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Total Revenue')
    plt.xticks(rotation=45, ha='right')
    fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 2 — Line chart: daily revenue trend
    daily_revenue = df.groupby('date')['revenue'].sum().sort_index()
    dates = daily_revenue.index
    revenues = daily_revenue.values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, revenues)
    ax.set_title('Daily Revenue Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Revenue')
    plt.xticks(rotation=45, ha='right')
    fig.savefig(f'{output_dir}/daily_revenue_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 3 — Horizontal bar chart: avg order value by payment method
    avg_order_by_payment = df.groupby('payment_method')['revenue'].mean()
    methods = avg_order_by_payment.index
    avg_values = avg_order_by_payment.values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(methods, avg_values)
    ax.set_title('Average Order Value by Payment Method')
    ax.set_xlabel('Average Order Value')
    ax.set_ylabel('Payment Method')
    fig.savefig(f'{output_dir}/avg_order_by_payment.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Run the full data pipeline end-to-end."""
    #Call load_data(DATA_PATH)
    df = load_data(DATA_PATH)
    #Call clean_data(df)
    df = clean_data(df)
    #Call add_features(df)
    df = add_features(df)
    #Call generate_summary(df) and print the results
    summary = generate_summary(df)
    for key, value in summary.items():
        print(f"{key}: {value}")
    #Call create_visualizations(df)
    create_visualizations(df)
    #Print "Pipeline complete."
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
