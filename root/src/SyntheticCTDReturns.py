# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:57:24 2024

@author: Diego
"""

import os
import pandas as pd
import pandas_datareader as web

class SyntheticCTD:
    
    def __init__(self) -> None:
        
        self.deliv_path  = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\BondDeliverableRisk"
        self.tsy_tickers = {
            "TU" : 2,
            "TY" : 10,
            "FV" : 5,
            "WN" : 30,
            "UXY": 20}
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.ctd_path  = os.path.join(self.data_path, "CTDReturn")
        
        if os.path.exists(self.ctd_path) == False: os.makedirs(self.ctd_path)
        
    def get_bond_deliverable(self) -> pd.DataFrame: 
        
        paths = ([
            os.path.join(self.deliv_path, file + ".parquet") 
            for file in self.tsy_tickers.keys()])
        
        df_combined = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split(" ").str[0]).
            pivot(index = ["date", "security"], columns = "variable", values = "value").
            dropna().
            rename(columns = {
                "CONVENTIONAL_CTD_FORWARD_FRSK": "duration",
                "FUT_EQV_CNVX_NOTL"            : "convexity"}).
            reset_index())
        
        return df_combined
    
    def get_synthetic_return_data(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ctd_path, "SyntheticCTD.parquet")
        try:
        
            if verbose == True: print("Trying to get return data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except:
        
            if verbose == True: print("Couldn't find CTD Synthetic data, generating it now")
            df_mat = (pd.DataFrame.from_dict(
                data    = self.tsy_tickers, 
                orient  = "index",
                columns = ["maturity"]).
                reset_index().
                rename(columns = {"index": "security"}).
                assign(
                    security = lambda x: x.security + "1",
                    maturity = lambda x: x.maturity.astype(int)))
            
            df_combined = (self.get_bond_deliverable().merge(
                right = df_mat, how = "inner", on = ["security"]))
            
            fred_tickers = [
                "DGS{}".format(self.tsy_tickers[sec]) 
                for sec in self.tsy_tickers.keys()]
            
            start_date, end_date = df_combined.date.min().date(), df_combined.date.max().date()
            df_fred = (web.DataReader(
                name        = fred_tickers, 
                data_source = "fred",
                start       = start_date,
                end         = end_date).
                reset_index().
                melt(id_vars = "DATE").
                rename(columns = {"DATE": "date"}).
                rename(columns = {
                    "variable": "yield_tenor", 
                    "value"   : "yield"}).
                assign(maturity = lambda x: x.yield_tenor.str.split("S").str[-1].astype(int)))
                
            df_out = (df_combined.merge(
                right = df_fred, how = "inner", on = ["date", "maturity"]))
        
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(yld_diff = lambda x: x.yld.diff()).
            dropna())
        
        return df_out
    
    def get_synthetic_rtn_calc(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.ctd_path, "SyntheticCTDRtn.parquet")
        try:
            
            if verbose == True: print("Trying to find CTD Return")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find CTD return data")
            df_out = (self.get_synthetic_return_data().rename(
                columns = {"yield": "yld"}).
                groupby("security").
                apply(self._get_diff).
                reset_index(drop = True).
                assign(
                    dur_rtn  = lambda x: x.yld_diff * x.duration,
                    cnvx_rtn = lambda x: (1 / 2) * (x.yld_diff ** 2) * x.convexity,
                    bnd_rtn  = lambda x: (x.yld / 360) -x.dur_rtn + x.cnvx_rtn).
                drop(columns = ["security", "maturity"]))

            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")


        
def main() -> None:

    SyntheticCTD().get_synthetic_return_data(verbose = True)
    SyntheticCTD().get_synthetic_rtn_calc(verbose = True)
    
if __name__ == "__main__": main()