# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:23:27 2024

@author: Diego
"""

import os
import requests
import subprocess
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

class GitHubCollector:
    
    def __init__(self) -> None:

        self.cusip_ts  = r"https://github.com/yieldcurvemonkey/CUSIP-Timeseries"
        self.cusip_set = r"https://github.com/yieldcurvemonkey/CUSIP-Set"
        
        self.root_path  = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path  = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.cusip_path = os.path.join(self.data_path, "CUSIPData")
        self.ts_path    = os.path.join(self.cusip_path, "TimeSeries")
        self.hist_path  = os.path.join(self.ts_path, "CTHistoricalYields")
    
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.cusip_path) == False: os.makedirs(self.cusip_path)
        if os.path.exists(self.ts_path) == False: os.makedirs(self.ts_path)
        if os.path.exists(self.hist_path) == False: os.makedirs(self.hist_path)

    def get_raw_cusip_ts(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.data_path, "CUSIP-Timeseries")
        if os.path.exists(file_path) == False: os.makedirs(file_path)
        
        if len(os.listdir(file_path)) == 0:

            subprocess.run(
                args   = ["git", "clone", self.cusip_ts],
                cwd    = self.data_path,
                check  = True,
                stdout = None,
                stderr = None)
            
        else: 
            if verbose == True: print("Have cusip time series data")
        
    def get_raw_historical_ct_yield(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.hist_path, "RawHistoricalYields.parquet")
        try:
            
            if verbose == True: print("Trying to find Historical Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            self.get_raw_cusip_ts()
            cusip_path  = os.path.join(self.data_path, "CUSIP-Timeseries")
            yield_paths = [
                os.path.join(cusip_path, path) 
                for path in os.listdir(cusip_path) 
                if path.split("_")[0] == "historical"]
            
            df_yield = (pd.concat([
                pd.read_json(path_or_buf = path).assign(quote = path.split("\\")[-1])
                for path in yield_paths]).
                assign(
                    Date  = lambda x: pd.to_datetime(x.Date).dt.date,
                    quote = lambda x: x.quote.str.split("_").str[-2]).
                rename(columns = {"Date": "date"}).
                melt(id_vars = ["date", "quote"]).
                rename(columns = {"variable": "tenor"}))
            
            df_tenor = (df_yield[
                ["tenor"]].
                drop_duplicates().
                assign(
                    tenor_front = lambda x: x.tenor.str.split("-").str[0].astype(int),
                    tenor_back  = lambda x: x.tenor.str.split("-").str[-1],
                    tenor_days  = lambda x: x.tenor_front * np.where(x.tenor_back == "Week", 7, 360)).
                drop(columns = ["tenor_front", "tenor_back"]))
        
            df_out = (df_tenor.merge(
                right = df_yield, how = "inner", on = ["tenor"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _get_zscore(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            roll_mean = lambda x: x.value.rolling(window = window).mean(),
            roll_std  = lambda x: x.value.rolling(window = window).std(),
            z_score   = lambda x: np.abs((x.value - x.roll_mean) / x.roll_std)).
            dropna().
            drop(columns = ["roll_mean", "roll_std", "value"]))
        
        return df_out
    
    def _interpolate_clean(self, df: pd.DataFrame, cutoff: int) -> pd.DataFrame: 
        
        replace_value = (df.query(
            "z_score < @cutoff").
            value.
            mean())
        
        df_out = (df.assign(
            replace_value = lambda x: np.where(x.z_score > cutoff, replace_value, x.value)))
        
        return df_out
    
    def _clean(
            self, 
            df: pd.DataFrame, 
            window_replace: int,
            long_window: int, 
            long_cutoff: int) -> pd.DataFrame: 
        
        print("Working on {}".format(df.name))
        
        df_tmp = (df.set_index(
            "date")
            [["bid", "eod", "mid", "offer"]].
            mean(axis = 1).
            to_frame(name = "avg").
            merge(right = df, how = "inner", on = ["date"]).
            assign(
                bid   = lambda x: x.bid.fillna(x.avg),
                eod   = lambda x: x.eod.fillna(x.avg),
                mid   = lambda x: x.mid.fillna(x.mid),
                offer = lambda x: x.offer.fillna(x.offer)).
            drop(columns = ["avg"]).
            melt(id_vars = ["date", "tenor_days", "tenor"]))
        
        df_first_zscore = (df_tmp.drop(
            columns = ["tenor_days", "tenor"]).
            groupby("variable").
            apply(self._get_zscore, long_window).
            reset_index(drop = True).
            merge(right = df_tmp, how = "inner", on = ["date", "variable"]))
        
        bad_dates = (df_first_zscore.query(
            "z_score > @long_cutoff").
            date.
            to_list())
        
        if len(bad_dates) != 0:
        
            df_first_clean = (df_first_zscore.query(
                "date == @bad_dates").
                groupby("date").
                apply(self._interpolate_clean, long_cutoff).
                reset_index(drop = True).
                drop(columns = ["z_score"]))
            
            df_good = (df_first_zscore.query(
                "date != @bad_dates").
                drop(columns = ["z_score"]))
            
            df_first_pass = (pd.concat(
                [df_good, df_first_clean]).
                assign(replace_value = lambda x: x.replace_value.fillna(x.value)))
            
        else: 
            
            df_first_pass = df_tmp.assign(replace_value = lambda x: x.value)
        
        return df_first_pass
    
    def get_cleaned_historical_ct_yield(self, verbose: bool = False) -> pd.DataFrame: 
        
        long_window: int = 200
        long_cutoff: int = 6
        window_replace: int = 100
        
        df_tmp = (self.get_raw_historical_ct_yield().pivot(
            index = ["date", "tenor_days", "tenor"], columns = "quote", values = "value").
            reset_index().
            assign(missing = lambda x: x.bid.fillna(1) + x.eod.fillna(1) + x.mid.fillna(1) + x.offer.fillna(1)))
        
        df_missing_all, df_missing_partial = df_tmp.query("missing == 4"), df_tmp.query("missing != 4")

        df_partial_clean = (df_missing_partial.drop(
            columns = ["missing"]).
            groupby("tenor").
            apply(self._clean, window_replace, long_window, long_cutoff).
            reset_index(drop = True))
        
        return df_partial_clean
    
    
def main() -> None: 
        
    
    df = GitHubCollector().get_raw_historical_ct_yield(verbose = True)
    df = GitHubCollector().get_cleaned_historical_ct_yield(verbose = True)
    
if __name__ == "__main__": main()