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
from itertools import combinations

class Spread(SyntheticCTD):
    
    def __init__(self) -> None: 
        
        super().__init__()
        
        self.inv_tsy_tickers = {v: k for k, v in self.tsy_tickers.items()}
        self.combos          = [
            ("DGS" + str(combo[0]), "DGS" + str(combo[1])) 
            for combo in list(combinations(list(self.inv_tsy_tickers.keys()), 2))
            if combo[0] < combo[1]]
        
    def _prep_rtn(self, df: pd.DataFrame, combo: tuple) -> pd.DataFrame: 
        
        df_namer = (pd.DataFrame(
            {
                "yield_tenor" : list(combo),
                "position"    : ["S", "L"]}))
             
        df_out = (df.merge(
            right = df_namer, how = "inner", on = ["yield_tenor"]).
            drop(columns = ["cvx_rtn", "bnd_rtn", "dur_rtn", "yield_tenor"]).
            rename(columns = {
                "convexity": "cvx",
                "duration" : "dur"}).
            melt(id_vars = ["date", "position"]).
            assign(variable = lambda x: x.position + x.variable).
            drop(columns = ["position"]).
            pivot(index = "date", columns = "variable", values = "value").
            sort_index().
            assign(
                lag_Ldur = lambda x: x.Ldur.shift(),
                lag_Sdur = lambda x: x.Sdur.shift(),
                spread   = combo[1] + "-" + combo[0],
                Lweight  = lambda x: x.lag_Sdur / (x.lag_Ldur + x.lag_Sdur),
                Sweight  = lambda x: x.lag_Ldur / (x.lag_Ldur + x.lag_Sdur),
                Ldur_rtn = lambda x: - (x.Ldur * x.Lyld_diff),
                Lcvx_rtn = lambda x: 0.5 * (x.Lyld_diff ** 2) * x.Lcvx,
                Sdur_rtn = lambda x: - (x.Sdur * x.Syld_diff),
                Scvx_rtn = lambda x: 0.5 * (x.Syld_diff ** 2) * x.Scvx,
                Lbnd_rtn = lambda x: (x.Lyld / 360) + x.Ldur_rtn + x.Lcvx_rtn,
                Sbnd_rtn = lambda x: (x.Syld / 360) + x.Sdur_rtn + x.Scvx_rtn).
            drop(columns = ["lag_Ldur", "lag_Sdur"]).
            dropna())
        
        return df_out
        
    def get_spread(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ctd_path, "SyntheticCTDSpreadRtn.parquet")
        try:
            
            if verbose == True: print("Trying to find CTD Spread Return")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
            df_prep = self.get_synthetic_rtn_calc()
            df_out  = (pd.concat([
                self._prep_rtn(df_prep, combo) 
                for combo in self.combos]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
 
def main() -> None: 
       
    df = Spread().get_spread(verbose = True)
    
if __name__ == "__main__": main()