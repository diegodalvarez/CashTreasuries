#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:35:52 2024

@author: diegoalvarez
"""

import os
import requests
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

class GitHubCollector:
    
    def __init__(self) -> None:

        self.url     = r"https://github.com/yieldcurvemonkey/Curve-Your-Enthusiasm-/tree/main/data"
        self.raw_url = r"https://raw.githubusercontent.com/yieldcurvemonkey/Curve-Your-Enthusiasm-/main/data"

        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawGitHubData")
        self.note_path = os.path.join(self.root_path, "notebooks")
        self.prep_path = os.path.join(self.data_path, "CleanedGitHubData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        if os.path.exists(self.note_path) == False: os.makedirs(self.note_path)
        if os.path.exists(self.prep_path) == False: os.makedirs(self.prep_path)
    
    def _get_github_xlsx_links(self) -> list:

        response = requests.get(self.url)
        soup     = BeautifulSoup(response.text, 'html.parser')
        elements = soup.find_all('div', class_='react-directory-truncate')
        
        files_ = [
            link.get("title") 
            for element in elements 
            for link in element.find_all("a", class_="Link--primary")]
                
        files = [file for file in files_ if file.split(".")[-1] == "xlsx"]
        links = ["{}/{}".format(self.raw_url, file) for file in files]
        
        return links
    
    def get_github_data(self, verbose: bool = False) -> None: 
        
        links = self._get_github_xlsx_links()
        for link in links:
            
            file_name = link.split("/")[-1]
            file_path = os.path.join(self.raw_path, file_name)
            
            try:
                
                if verbose == True: print("Trying to find {}".format(file_name))
                df_out = pd.read_excel(io = file_path)
                if verbose == True: print("Found data\n")
                
            except: 
            
                if verbose == True: print("Couldn't find data, now collecting")
                df_out = pd.read_excel(io = link)
                df_out.to_excel(excel_writer = file_path)
                if verbose == True: print("Saving data\n")
                
    def get_filtered_auction(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.prep_path, "AuctionInfo.parquet")
        try:
            
            if verbose == True: print("Trying to find Treasury Auction Data")
            df_key = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it now")
            
            read_in = os.path.join(self.raw_path, r"filtered_auctions.xlsx")
            df_key  =  (pd.read_excel(
                io = read_in, index_col = [0,1]).
                reset_index(drop = True).
                assign(
                    auction_date = lambda x: pd.to_datetime(x.auction_date).dt.date,
                    issue_date   = lambda x: pd.to_datetime(x.issue_date).dt.date))
            
            if verbose == True: print("Saving data\n")
            df_key.to_parquet(path = file_path, engine = "pyarrow")
    
        return df_key
    
    def _get_historical_auctions(self) -> pd.DataFrame:
    
        melt_cols = list(self.get_filtered_auction().columns) + ["record_date"] + ["security_term"]
        tmp_path  = os.path.join(self.raw_path, r"historical_auctions.xlsx")
        
        df_historical_auctions = (pd.read_excel(
            io = tmp_path, index_col = 0).
            melt(id_vars = melt_cols).
            dropna().
            assign(
                auction_date = lambda x: pd.to_datetime(x.auction_date).dt.date,
                issue_date   = lambda x: pd.to_datetime(x.issue_date).dt.date,
                record_date  = lambda x: pd.to_datetime(x.record_date).dt.date))
    
        return df_historical_auctions
    
    def _get_historical_treasury_auctions(self) -> pd.DataFrame: 
    
        melt_cols = list(self.get_filtered_auction().columns) + ["record_date"] + ["security_term"]
        tmp_path = os.path.join(self.raw_path, r"Histroical_Treasury_Auctions.xlsx")
        
        df_historical = (pd.read_excel(
            io = tmp_path, index_col = 0).
            melt(id_vars = melt_cols).
            dropna().
            assign(
                auction_date = lambda x: pd.to_datetime(x.auction_date).dt.date,
                issue_date   = lambda x: pd.to_datetime(x.issue_date).dt.date,
                record_date  = lambda x: pd.to_datetime(x.record_date).dt.date))
    
        return df_historical
    
    def combine_auction_data(self) -> pd.DataFrame:
        
        merge_cols = list(self.get_filtered_auction().columns) + ["record_date"] + ["security_term"] + ["variable"]
        
        df_left = (self._get_historical_auctions().rename(
            columns = {"value": "from_historical_auctions"}))
        
        df_right = (self._get_historical_treasury_auctions().rename(
            columns = {"value": "from_historical_treasury_auctions"}))
        
        df_combined = (df_left.merge(
            right = df_right, how = "outer", on = merge_cols))
        
        df_out = (df_combined.assign(
            new_val = lambda x: np.where(
                x.from_historical_auctions != x.from_historical_treasury_auctions,
                np.nan,
                x.from_historical_treasury_auctions)).
            assign(new_val = lambda x: x.new_val.fillna(x.from_historical_auctions)).
            assign(new_val = lambda x: x.new_val.fillna(x.from_historical_treasury_auctions).astype(str)).
            drop(columns = ["from_historical_treasury_auctions", "from_historical_auctions"]).
            rename(columns = {"new_val": "value"}))
    
        return df_out
        
    def get_historical_auction(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.prep_path, "HistoricalAuction.parquet")
        try:
            
            if verbose == True: print("Trying to find Treasury Auction Results Data")
            df_auct = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data, collecting it")
            df_auct = self.combine_auction_data()
            if verbose == True: print("Saving data\n")
            df_auct.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_auct
    
    def combine_observed_treasuries(self) -> pd.DataFrame:
    
        files = [
            os.path.join(self.raw_path, file) 
            for file in os.listdir(self.raw_path) 
            if len(file.split("_")) == 6]
        
        df_observed = (pd.concat([
            pd.read_excel(io = file, index_col = 0).assign(
                group = file.split("/")[-1].split(".")[0].split("usts_")[-1])
            for file in files]).
            melt(id_vars = ["Action", "Description", "group", "last_updated"]).
            dropna().
            drop_duplicates())
        
        df_out = (df_observed.groupby(
            ["Action", "Description", "group", "last_updated", "variable"]).
            head(1).
            pivot(
                index   = ["Action", "Description", "last_updated", "variable"], 
                columns = "group", 
                values  = "value").
            reset_index().
            assign(new_val = 
                   lambda x: np.where(
                       (
                           (x.auctioned_after_2000 == x.auctioned_after_2005) & 
                           (x.auctioned_after_2005 == x.auctioned_after_2010)),
                       x.auctioned_after_2000,
                       np.nan)).
            assign(new_val = lambda x: x.new_val.fillna(x.auctioned_after_2000)).
            assign(new_val = lambda x: x.new_val.fillna(x.auctioned_after_2005)).
            assign(new_val = lambda x: x.new_val.fillna(x.auctioned_after_2010)).
            drop(columns = ["auctioned_after_2000", "auctioned_after_2005", "auctioned_after_2010"]))
    
        return df_out
                

def main() -> None: 

    GitHubCollector().get_github_data(verbose = True)
    GitHubCollector().get_filtered_auction(verbose = True)
    GitHubCollector().get_historical_auction(verbose = True)
    
if __name__ == "__main__": main()