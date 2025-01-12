# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:22:56 2025

@author: Diego
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SyntheticCTDReturns import SyntheticCTD
from itertools import permutations
from tqdm import tqdm

class CTDFly(SyntheticCTD):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.inv_tsy_tickers = {v: k for k, v in self.tsy_tickers.items()}
        self.combos          = [
            ("DGS" + str(combo[0]), "DGS" + str(combo[1]), "DGS" + str(combo[2])) for combo in 
            list(permutations(list(self.inv_tsy_tickers.keys()), 3))
            if combo[0] < combo[1] < combo[2]]
        
    def _get_fly(self, df: pd.DataFrame, combo: tuple) -> pd.DataFrame:
             
        fly_tickers = list(combo)
        renamer     = {
            combo[0]: "short",
            combo[1]: "bullet",
            combo[2]: "long"}
        
        df_namer = (pd.DataFrame.from_dict(
            data    = renamer, 
            orient  = "index",
            columns = ["pos"]).
            reset_index().
            rename(columns = {"index": "yield_tenor"}))
        
        df_weight = (df.query(
            "yield_tenor == @fly_tickers").
            merge(right = df_namer, how = "inner", on = ["yield_tenor"]).
            pivot(index = "date", columns = "pos", values = "duration").
            dropna().
            assign(
                short_weight  = lambda x: 0.5 * (x.long - x.bullet) / (x.long - x.short),
                long_weight   = lambda x: 0.5 * (x.bullet - x.short) / (x.long - x.short),
                bullet_weight = 0.5).
            drop(columns = ["bullet", "long", "short"]).
            shift().
            reset_index().
            dropna().
            assign(fly = lambda x: 
                   combo[0].replace("DGS", "") + 
                   combo[1].replace("DGS", "s") + 
                   combo[2].replace("DGS", "s") + "s"))
        
        df_out = (df.query(
            "yield_tenor == @fly_tickers").
            merge(right = df_namer, how = "inner", on = ["yield_tenor"]).
            drop(columns = ["yield_tenor"]).
            melt(id_vars = ["date", "pos"]).
            assign(pos = lambda x: x.pos + "_val").
            pivot(index = ["date", "variable"], columns = "pos", values = "value").
            dropna().
            reset_index().
            merge(right = df_weight, how = "inner", on = ["date"]))   
        
        return df_out
    
    def get_flies(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ctd_path, "SyntheticCTDFlyRtn.parquet")
        try:
        
            if verbose == True: print("Trying to get CTD Steepener return data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
        
            if verbose == True: print("Couldn't find data, generating it now")
        
            df     = self.get_synthetic_rtn_calc()
            df_out = (pd.concat([
                self._get_fly(df, combo)
                for combo in tqdm(self.combos, desc = "Calculting Flies")]))
        
            if verbose == True: print("\nSaving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
def main() -> None:
    
    df = CTDFly().get_flies(verbose = True)
    
if __name__ == "__main__": main()