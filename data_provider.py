import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

CACHE_FILE = "price_cache.parquet"

def fetch_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches adjusted close prices for a list of tickers, using a local parquet cache
    to minimize yfinance calls.

    Mathematical/Data Schema:
    - Returns: pandas.DataFrame
    - Index: pandas.DatetimeIndex (frequency: business days, though yfinance may have gaps)
    - Columns: Ticker symbols (str)
    - Values: Adjusted Close prices (float64) representing the asset value $S_t$.
    - Units: Base currency of the asset (usually USD).

    Caching Logic:
    1. Loads `price_cache.parquet` if it exists.
    2. Identifies which tickers and date ranges are missing from the cache.
    3. Fetches *only* the missing data from `yfinance`.
    4. Merges the new data with the cached data and saves back to `price_cache.parquet`.
    5. Returns the requested subset of data.
    """
    tickers = [t.upper() for t in tickers]
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Ensure end_date includes the full day if it's today
    if end_dt.date() == datetime.today().date():
         end_dt = end_dt + timedelta(days=1)

    cached_df = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_parquet(CACHE_FILE)
        except Exception as e:
            print(f"Warning: Could not read cache file '{CACHE_FILE}': {e}. Proceeding without cache.")
            cached_df = pd.DataFrame()

    tickers_to_fetch = []
    fetch_start = start_dt
    fetch_end = end_dt

    if not cached_df.empty:
        # Check which tickers are missing entirely
        cached_tickers = cached_df.columns.tolist()
        missing_tickers = [t for t in tickers if t not in cached_tickers]
        tickers_to_fetch.extend(missing_tickers)

        # Check if the requested date range is fully covered for existing tickers
        cached_start = cached_df.index.min()
        cached_end = cached_df.index.max()

        date_extension_needed = False
        
        if start_dt < cached_start:
            fetch_start = start_dt
            date_extension_needed = True
        
        if end_dt > cached_end + timedelta(days=1): # yfinance end_date is exclusive, add tolerance
             fetch_end = end_dt
             date_extension_needed = True
             
        if date_extension_needed:
             # If we need to extend dates, we must refetch the existing tickers for the missing periods
             # For simplicity in this robust caching model, if dates are outside the cached range,
             # we fetch the *entire* requested range for *all* requested tickers to ensure data alignment.
             # A more complex range-diffing algorithm could be implemented, but this guarantees 
             # consistency and `yfinance` handles bulk requests efficiently.
             tickers_to_fetch = tickers
             fetch_start = start_dt
             fetch_end = end_dt
             
    else:
        tickers_to_fetch = tickers

    new_data = pd.DataFrame()
    if tickers_to_fetch:
        print(f"Fetching data from yfinance for: {tickers_to_fetch} from {fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')}")
        try:
            # yfinance returns a MultiIndex if multiple tickers, but a single level if one ticker
            df = yf.download(tickers_to_fetch, start=fetch_start, end=fetch_end, auto_adjust=False)
            
            if len(tickers_to_fetch) == 1:
                # yf.download returns a simple dataframe for 1 ticker. We extract 'Adj Close'
                if 'Adj Close' in df.columns:
                    new_data = df[['Adj Close']].copy()
                    new_data.columns = [tickers_to_fetch[0]]
            elif df.columns.nlevels > 1:
                # Multiple tickers return a MultiIndex (Price, Ticker)
                if 'Adj Close' in df.columns.levels[0]:
                    new_data = df['Adj Close'].copy()
                elif 'Close' in df.columns.levels[0]:
                    new_data = df['Close'].copy() # Fallback if Adj Close is missing
                    
            if not new_data.empty:
                # Ensure index is timezone-naive for parquet compatibility
                if new_data.index.tz is not None:
                     new_data.index = new_data.index.tz_localize(None)
                     
                # Merge with cached data
                if not cached_df.empty and not date_extension_needed:
                    # We only fetched new tickers for the same date range, or we are merging new tickers
                    cached_df = pd.concat([cached_df, new_data], axis=1)
                elif not cached_df.empty and date_extension_needed:
                    # We fetched new date ranges. Combine and deduplicate
                    combined_df = pd.concat([cached_df, new_data], axis=1)
                    # Support modern pandas >= 2.1 which deprecated axis=1 in groupby.
                    # Transpose, group by index (which were columns), take first, transpose back.
                    cached_df = combined_df.T.groupby(level=0).first().T
                else:
                    cached_df = new_data
                
                # Sort index to guarantee temporal order
                cached_df.sort_index(inplace=True)
                
                # Save back to cache
                cached_df.to_parquet(CACHE_FILE)
                
        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            raise

    # Return only the requested subset
    if not cached_df.empty:
        # Filter by requested date range
        mask = (cached_df.index >= pd.to_datetime(start_date)) & (cached_df.index <= pd.to_datetime(end_date))
        result_df = cached_df.loc[mask]
        
        # Filter by requested tickers, keeping only those that exist
        available_tickers = [t for t in tickers if t in result_df.columns]
        result_df = result_df[available_tickers]
        
        # Forward fill and backward fill any isolated missing values
        result_df = result_df.ffill().bfill()
        
        # Ensure float64 as requested in rules
        result_df = result_df.astype('float64')
        
        return result_df
    
    return pd.DataFrame()
