# enhanced_fetch_data.py
import yfinance as yf
from datetime import timedelta, date
import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker: str, prediction_date: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch historical stock data up to 1 day before the prediction date.
   
    Parameters:
    - ticker: Stock symbol (e.g., "RELIANCE.NS")
    - prediction_date: String in 'YYYY-MM-DD' format
    - lookback_days: Number of past days to use for training
   
    Returns:
    - DataFrame with historical stock data
    """
    try:
        end_date = pd.to_datetime(prediction_date) - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days)
       
        # Add progress indicator (with safety check for streamlit context)
        progress_bar = None
        status_text = None
        
        try:
            # Check if we're in a Streamlit context more reliably
            if hasattr(st, '_main') and st._main is not None:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Downloading data for {ticker}...")
                progress_bar.progress(25)
        except (AttributeError, RuntimeError):
            # Not in streamlit context, use print instead
            print(f"Downloading data for {ticker}...")
       
        # Download data with error handling
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if progress_bar:
            progress_bar.progress(50)
       
        if df.empty:
            raise ValueError(f"No data found for {ticker}. Please check the symbol or try a different date range.")
       
        # Debug: Print available columns
        print(f"Available columns for {ticker}: {df.columns.tolist()}")
        
        # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove ticker level
        
        # Clean and process data - Handle missing 'Adj Close' column
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = []
        
        # Check which columns are available and add them
        for col in required_columns:
            if col in df.columns:
                available_columns.append(col)
        
        # Handle Adj Close specifically
        if 'Adj Close' in df.columns:
            available_columns.append('Adj Close')
        elif 'Close' in df.columns:
            # If Adj Close is not available, create it as a copy of Close
            df['Adj Close'] = df['Close']
            available_columns.append('Adj Close')
        else:
            raise ValueError(f"No Close or Adj Close column found in data for {ticker}")
        
        if progress_bar:
            progress_bar.progress(75)
        
        # Select only available columns
        df = df[available_columns].copy()
        
        # Reset index to get Date as a column
        df.reset_index(inplace=True)
        
        # Standardize column names to lowercase
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        # Only rename columns that exist
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=existing_mapping, inplace=True)
        
        # Handle missing values - FIX: Use proper method for checking NaN values
        df = df.dropna()
       
        if len(df) < 50:
            raise ValueError(f"Insufficient data points ({len(df)}). Need at least 50 days of data.")
       
        # Ensure we have the target column (prioritize adj_close, fallback to close)
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']
        
        # Ensure date column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        if progress_bar:
            progress_bar.progress(100)
            status_text.text("✅ Data downloaded successfully!")
            
            # Clear progress indicators after a short delay
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        else:
            print("✅ Data downloaded successfully!")
       
        print(f"Final dataframe shape: {df.shape}")
        print(f"Final columns: {df.columns.tolist()}")
        
        return df
       
    except Exception as e:
        # Clean up progress indicators on error
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")

def prepare_train_test_split(df: pd.DataFrame, 
                           test_size: float = 0.2, 
                           time_based_split: bool = True,
                           target_column: str = 'adj_close') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets
    
    Parameters:
    - df: DataFrame with stock data
    - test_size: Proportion of data for testing (0.2 = 20%)
    - time_based_split: If True, use chronological split (recommended for time series)
    - target_column: Name of the target column (default: 'adj_close')
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    
    # Define feature columns (exclude date and target)
    exclude_cols = ['date', target_column]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_columns]
    y = df[target_column]
    
    if time_based_split:
        # Time-based split (recommended for stock data)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Time-based split:")
        print(f"Training data: {len(X_train)} samples (index {X_train.index.min()} to {X_train.index.max()})")
        print(f"Testing data: {len(X_test)} samples (index {X_test.index.min()} to {X_test.index.max()})")
        
    else:
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Random split:")
        print(f"Training data: {len(X_train)} samples")
        print(f"Testing data: {len(X_test)} samples")
    
    print(f"Feature columns: {feature_columns}")
    print(f"Target column: {target_column}")
    
    return X_train, X_test, y_train, y_test

def get_data_with_split(ticker: str, 
                       prediction_date: str, 
                       lookback_days: int = 365,
                       test_size: float = 0.2,
                       time_based_split: bool = True,
                       target_column: str = 'adj_close') -> dict:
    """
    Complete pipeline: fetch data and split into train/test
    
    Parameters:
    - ticker: Stock symbol
    - prediction_date: Date string in 'YYYY-MM-DD' format
    - lookback_days: Number of historical days
    - test_size: Proportion for test set (0.2 = 20%)
    - time_based_split: Use chronological split vs random
    - target_column: Column to predict (default: 'adj_close')
    
    Returns:
    Dictionary containing all data splits and metadata
    """
    
    print(f"Starting complete data pipeline for {ticker}...")
    
    # Step 1: Fetch raw data
    print("Step 1: Fetching stock data...")
    raw_data = fetch_stock_data(ticker, prediction_date, lookback_days)
    
    # Step 2: Train-test split
    print("Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        raw_data, test_size, time_based_split, target_column
    )
    
    # Create result dictionary
    result = {
        'raw_data': raw_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': X_train.columns.tolist(),
        'target_column': target_column,
        'ticker': ticker,
        'prediction_date': prediction_date,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'split_type': 'time_based' if time_based_split else 'random'
    }
    
    print("✅ Data pipeline completed successfully!")
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return result

def validate_stock_symbol(ticker: str) -> bool:
    """
    Validate if a stock symbol exists and has data
    """
    try:
        # Test download a small amount of data
        test_data = yf.download(ticker, period="5d", progress=False)
        return not test_data.empty
    except Exception:
        return False

def get_stock_info(ticker: str) -> dict:
    """
    Get basic stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
       
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'INR')
        }
    except Exception as e:
        print(f"Error getting stock info: {e}")
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 'N/A',
            'currency': 'INR'
        }

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional data cleaning function to handle edge cases
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove any completely empty rows
    df_clean = df_clean.dropna(how='all')
    
    # For numerical columns, replace inf/-inf with NaN then drop
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN in critical columns
    critical_cols = ['close', 'adj_close'] if 'adj_close' in df_clean.columns else ['close']
    existing_critical_cols = [col for col in critical_cols if col in df_clean.columns]
    
    if existing_critical_cols:
        df_clean = df_clean.dropna(subset=existing_critical_cols)
    
    return df_clean

def test_complete_pipeline():
    """
    Test the complete pipeline with train-test split
    """
    test_ticker = "RELIANCE.NS"
    try:
        print(f"Testing complete pipeline for {test_ticker}...")
        
        # Test the complete pipeline
        result = get_data_with_split(
            ticker=test_ticker,
            prediction_date=str(date.today()),
            lookback_days=100,
            test_size=0.2,
            time_based_split=True
        )
        
        print("\n" + "="*50)
        print("PIPELINE TEST RESULTS")
        print("="*50)
        
        # Display results
        print(f"✅ Pipeline completed successfully!")
        print(f"Ticker: {result['ticker']}")
        print(f"Target column: {result['target_column']}")
        print(f"Split type: {result['split_type']}")
        print(f"Raw data shape: {result['raw_data'].shape}")
        print(f"Features: {len(result['feature_columns'])}")
        print(f"Training samples: {result['train_size']}")
        print(f"Testing samples: {result['test_size']}")
        
        print("\nFeature columns:")
        for i, col in enumerate(result['feature_columns'], 1):
            print(f"  {i}. {col}")
        
        print("\nTraining data sample:")
        print(result['X_train'].head())
        print("\nTarget values sample:")
        print(result['y_train'].head())
        
        # Data quality checks
        print(f"\nData quality check:")
        print(f"- Training set missing values: {result['X_train'].isnull().sum().sum()}")
        print(f"- Testing set missing values: {result['X_test'].isnull().sum().sum()}")
        print(f"- Target missing values (train): {result['y_train'].isnull().sum()}")
        print(f"- Target missing values (test): {result['y_test'].isnull().sum()}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Additional utility functions
def get_common_indian_stocks():
    """
    Returns a list of common Indian stock symbols for testing
    """
    return [
        "RELIANCE.NS",
        "TCS.NS", 
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "HINDUNILVR.NS",
        "LT.NS"
    ]

def batch_validate_symbols(symbols: list) -> dict:
    """
    Validate multiple stock symbols at once
    """
    results = {}
    for symbol in symbols:
        print(f"Validating {symbol}...")
        results[symbol] = validate_stock_symbol(symbol)
    return results

if __name__ == "__main__":
    print("Testing enhanced fetch_data.py module...")
    print("="*60)
    
    # Test complete pipeline
    result = test_complete_pipeline()
    
    if result:
        print("\n" + "="*60)
        print("Testing additional validation...")
        common_stocks = get_common_indian_stocks()[:3]  # Test first 3
        validation_results = batch_validate_symbols(common_stocks)
        print("Validation results:", validation_results)