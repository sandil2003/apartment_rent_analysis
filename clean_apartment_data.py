import pandas as pd
import numpy as np

def clean_numeric_column(series):
    """Clean numeric columns that have periods instead of decimal points"""
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.replace(',', '')
        parts = series.str.split('.')
        series = parts.str[0] + '.' + parts.str[1].fillna('0')
        series = pd.to_numeric(series, errors='coerce')
    return series

def clean_apartment_data(input_file, output_file):
    """Clean the apartment market prices dataset"""
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}\n")
    
    print("Cleaning numeric columns...")
    numeric_cols = [
        'Tract Median Apartment Contract Rent per Square Foot',
        'Tract Median Apartment Contract Rent per Unit',
        'Year over Year Change in Rent per Square Foot',
        'Year over Year Change in Rent per Unit',
        'PROPERTIES',
        'Shape__Area',
        'Shape__Length'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
            print(f"  Cleaned: {col}")
    
    print("\nReplacing zeros with NaN for meaningful columns...")
    zero_to_nan_cols = [
        'Tract Median Apartment Contract Rent per Square Foot',
        'Tract Median Apartment Contract Rent per Unit'
    ]
    
    for col in zero_to_nan_cols:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
            print(f"  Replaced zeros: {col}")
    
    print("\nHandling missing values...")
    print(f"Missing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\nCleaning categorical columns...")
    if 'Cost Category' in df.columns:
        df['Cost Category'] = df['Cost Category'].fillna('Unknown')
    
    if 'Year over Year Change in Rent Category' in df.columns:
        df['Year over Year Change in Rent Category'] = df['Year over Year Change in Rent Category'].fillna('Unknown')
    
    print("\nRemoving duplicate rows...")
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  Removed {duplicates} duplicate rows")
    
    print("\nData types after cleaning:")
    print(df.dtypes)
    
    print(f"\nCleaned shape: {df.shape}")
    print("\nBasic statistics of cleaned numeric columns:")
    print(df[numeric_cols].describe())
    
    print(f"\nSaving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("Data cleaning complete!")
    print(f"Original rows: {pd.read_csv(input_file).shape[0]}")
    print(f"Cleaned rows: {df.shape[0]}")
    print(f"Output file: {output_file}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    input_file = 'Apartment_Market_Prices new.csv'
    output_file = 'Apartment_Market_Prices_cleaned.csv'
    
    cleaned_df = clean_apartment_data(input_file, output_file)
