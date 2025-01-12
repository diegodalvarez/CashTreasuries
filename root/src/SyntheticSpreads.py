#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:14:25 2025

@author: diegoalvarez
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SyntheticCTDReturns import SyntheticCTD
from itertools import permutations
from tqdm import tqdm

class Spread(SyntheticCTD):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.inv_tsy_tickers = {v: k for k, v in self.tsy_tickers.items()}
        self.combos          = [
            ("DGS" + str(combo[0]), "DGS" + str(combo[1])) 
            for combo in list(permutations(list(self.inv_tsy_tickers.keys()), 2))
            if combo[0] < combo[1]] 
    
    def _get_rtn(self, df: pd.DataFrame, combo: tuple) -> pd.DataFrame: 
        
        combo_dict = {
            "long" : combo[0],
            "short": combo[1]}
        
        renamer = {
            "long weight" : "long_weight",
            "short weight": "short_weight",
            "long"        : "long_val",
            "short"       : "short_val"}
        
        tenors = list(combo_dict.values())
        
        df_namer = (pd.DataFrame.from_dict(
            data    = combo_dict, 
            orient  = "index",
            columns = ["yield_tenor"]).
            reset_index().
            rename(columns = {"index": "position"}))
        
        df_tmp = (df.query(
            "yield_tenor == @tenors").
            merge(right = df_namer, how = "inner", on = ["yield_tenor"]))
        
        df_inv_dur = (df_tmp.pivot(
            index = "date", columns = "yield_tenor", values = "duration").
            apply(lambda x: 1 / x).
            shift().
            dropna().
            reset_index().
            melt(id_vars = "date"))
        
        df_out = (df_inv_dur.drop(
            columns = ["yield_tenor"]).
            groupby("date").
            agg("sum").
            rename(columns = {"value": "cum_value"}).
            merge(right = df_inv_dur, how = "inner", on = ["date"]).
            assign(weight = lambda x: x.value / x.cum_value)
            [["date", "yield_tenor", "weight"]].
            merge(right = df_tmp, how = "inner", on = ["date", "yield_tenor"]).
            drop(columns = ["yield_tenor"]).
            melt(id_vars = ["date", "position"]).
            assign(tmp = lambda x: x.position + " " + x.variable).
            pivot(index = "date", columns = "tmp", values = "value").
            reset_index().
            melt(id_vars = ["date", "long weight", "short weight"]).
            assign(
                pos = lambda x: x.tmp.str.split(" ").str[0],
                var = lambda x: x.tmp.str.split(" ").str[1]).
            pivot(index = ["date", "long weight", "short weight", "var"], columns = "pos", values = "value").
            reset_index().
            rename(columns = renamer).
            assign(steepener = combo[0].replace("DGS", "") + "s" + combo[1].replace("DGS", "") + "s steepener"))
        
        return df_out
    
    def get_spread(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.ctd_path, "SyntheticCTDSteepenerRtn.parquet")
        try:
        
            if verbose == True: print("Trying to get CTD Steepener return data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
        
            if verbose == True: print("Couldn't find data, generating it now")
        
            df     = self.get_synthetic_rtn_calc()
            df_out = (pd.concat([
                self._get_rtn(df, combo)
                for combo in tqdm(self.combos, desc = "Processing Pair")]))
            
            if verbose == True: print("\nSaving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
 
def main() -> None: 
       
    df = Spread().get_spread(verbose = True)
    
if __name__ == "__main__": main()