"""Utility functions."""

# Import packages
from collections import OrderedDict
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Union, Optional

import imageio
import numpy as np
import pandas as pd

## I/O
def load_json(filepath:str) -> Union[List, Dict]:
    """Load JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def create_dirs(dirs: Union[str, List[str]]) -> None:
    """Create single/multiple directories."""
    dirs = dirs if isinstance(dirs, list) else [dirs]
    for path in dirs:
        os.makedirs(path, exist_ok=True)

def increment_folder_name(folder: str) -> str:
    """If folder currently exists, increment with version number."""
    if os.path.isdir(folder):
        version = 2
        versioned_folder = f'{folder} (v{version})'
        while os.path.isdir(versioned_folder):
            version += 1
            versioned_folder =  f'{folder} (v{version})'
        return versioned_folder
    else:
        return folder

def list_files_within_year_range(
    data_dir: str,
    file_prefix: str,
    start_year: int,
    end_year: int,
) -> List[str]:
    """Create list of file names within start and end year."""
    year_2dgt_list = [str(x)[2:] for x in np.arange(start_year, end_year + 1)]
    files_prefix_list = tuple(f'{file_prefix}{yr}' for yr in year_2dgt_list)
    return [x for x in os.listdir(data_dir) if x.startswith(files_prefix_list)]

def import_and_concat_csv_data(
    data_dir: str,
    files_list: List[str],
    cols_list: Optional[List[str]] = None,
    filter_ids: Optional[pd.Series] = None,
    id_var: Optional[str] = None,
) -> pd.DataFrame:
    """Import all CSV files from files list, filter on id variable, and concatenate."""
    data_list = []
    for file in files_list:
        print(f'Importing {file}.')
        try:
            data = pd.read_csv(os.path.join(data_dir, file), usecols=cols_list)
            if filter_ids is not None:
                data = data.loc[data[id_var].isin(filter_ids)]
            data_list.append(data)
        except FileNotFoundError:
            print(f'File "{file}" does not exist at specified path and was not merged.')
            continue
    return pd.concat(data_list, ignore_index=True)

## Logging
def create_log(path: str) -> None:
    sys.stdout = open(path, 'w')
    last_run_date = datetime.now().strftime('%H:%M:%S on %d %b %Y')
    print(f'Last run at {last_run_date}. \n')

def format_current_time() -> str:
    return datetime.now().strftime('%H:%M:%S')

## Memory management
def check_memory_usage(df: pd.DataFrame) -> float:
    """Check memory usage (in MB) for dataframe object."""
    return df.memory_usage(index=False, deep=True).sum() / (1024**2)

def downcast_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric (float and integer) columns to smallest type."""
    for numeric_type in ['float', 'integer']:
        downcast_cols = list(df.select_dtypes(include=[numeric_type]).columns)
        for col in downcast_cols:
            start_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], downcast=numeric_type)
            end_dtype = df[col].dtype
            if start_dtype != end_dtype:
                print(f'Downcast column {col} from {start_dtype} to {end_dtype}.')
    return df

def convert_categorical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to categorical datatype."""
    object_cols = list(df.select_dtypes(include=['object']).columns)
    df.loc[:, object_cols] = df.loc[:, object_cols].astype('category')
    return df

## Data cleaning
def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    results = OrderedDict()
    for col in sorted(df.columns):
        is_null_bool = df[col].isnull()
        is_duplicated_bool = df[col].duplicated(keep=False)
        results[col] = {
            'column_dtype':df[col].dtype,
            'number_of_dtypes_in_col_values':df[col].map(type).nunique(),
            'null_count':is_null_bool.sum(),
            'null_share':is_null_bool.mean(),
            'number_of_unique_values':df[col].nunique(),
            'duplicated_count':is_duplicated_bool.sum(),
            'duplicated_share':is_duplicated_bool.mean(),
        }
    return pd.DataFrame(results)

def convert_datetime_cols(df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame: 
    """Convert datetime columns to pandas datetime dtype."""
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col]).replace(pd.to_datetime('1910-01-01'), pd.NaT)
        print(f'Column "{col}" dtype: {df[col].dtype}')
    return df

#TODO: benchmark this function against pandas dropna
def drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with entirely null values (in memory efficient way)."""
    preserve_cols_list = []
    for col in df.columns:
        if not df[col].isnull().all():
            preserve_cols_list.append(col)
    return df.loc[:, preserve_cols_list]

## Data merging
def prepare_for_join(df: pd.DataFrame, merge_cols: List[str]) -> pd.DataFrame:
    """Prepare data for join by:
        - Dropping null columns
        - Converting datetime columns to pandas datetime dtype
        - Downcasting numeric columns
        - Converting categorical columns to pandas category dtype
        - Dropping duplicates on merge columns
        - Sorting values on merge columns
        - Setting index on merge columns
    """
    df = drop_null_columns(df)
    df = convert_datetime_cols(df, datetime_cols=[]) #TODO
    df = convert_categorical_cols(df)
    df = downcast_numeric_cols(df)
    df = df.drop_duplicates(merge_cols).sort_values(merge_cols).set_index(merge_cols)
    return df
    
class SmartMerger():
    """A SmartMerger class that checks conditions for successful merge BEFORE attempting,
    and returns clear error messages if conditions are not met.
    
    Attributes
        df1: merge data formatted as pandas DataFrame
        df2: merge data formatted as pandas DataFrame
        merge_cols: list of merge column names (i.e., pd.merge "on=" argument)
        merge_type: pd.merge "how=" argument (e.g., "left", "inner", "outer")
        validate: pd.merge "validate=" argument (e.g., "1:1", "m:1", "m:m")
        drop_dups: boolean indicator for whether duplicate values should be 
            dropped before merging (default == True)
        keep_dups: pd.drop_duplicates "keep=" argument (e.g., "first", "False")

    Methods
        merge(): checks conditions for successful merge; if met, merges data
        handle_duplicates(): checks the share of duplicates in merge columns
            subset and drops duplicates if drop_dups attribute is True
        check_col_exists(): checks if column exists in both df1 and df2
        check_common_values(): checks if column has common values in both df1 and df2
        check_same_dtype(): checks that column has the EXACT same dtype in both
            df1 and df2, and that column does not have multiple dtypes in values
    """

    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        merge_cols: List[str],
        merge_type: str,
        validate: str,
        drop_dups: bool = True,
        keep_dups: str = 'first',
    ) -> None:
        """Instantiate class with data and merge parameters."""
        self.df1 = df1
        self.df2 = df2
        self.merge_cols = merge_cols
        self.merge_type = merge_type
        self.validate = validate
        self.drop_dups = drop_dups
        self.keep_dups = keep_dups

    def merge(self):
        """Check conditions for successful merge and merge data if all conditions are met."""
        # Check for errors that would prevent merge
        for col in self.merge_cols:
            col_exist_errors = self.check_col_exists(col)
            dtype_errors = self.check_same_dtype(col)
            common_val_errors = self.check_common_values(col)
        errors_list = col_exist_errors + dtype_errors + common_val_errors
        if errors_list:
            print('\nErrors preventing merge: \n{}'.format('\n'.join(errors_list)))   
            return None
        else:
            # Handle duplicates and merge data
            print('All conditions met for successful merge. Checking/handling duplicates.')
            self.handle_duplicates()
            return pd.merge(
                self.df1,
                self.df2,
                how=self.merge_type,
                on=self.merge_cols,
                validate=self.validate,
            )

    def handle_duplicates(self):
        """Check share of duplicates in merge column values and drop if drop duplicates
        parameter set to TRUE."""
        # Check duplicate share
        df1_dup_share = self.df1.duplicated(subset=self.merge_cols, keep=False).mean()
        df2_dup_share = self.df2.duplicated(subset=self.merge_cols, keep=False).mean()
        print(f'Share of duplicate obs. in merge columns for df1: {round(df1_dup_share, 2)}')
        print(f'Share of duplicate obs. in merge columns for df2: {round(df2_dup_share, 2)}')
        if self.drop_dups:
            # Drop duplicate values in merge columns
            if df1_dup_share > 0:
                self.df1 = self.df1.drop_duplicates(
                    subset=self.merge_cols,
                    keep=self.keep_dups,
                ).reset_index(drop=True)
            if df2_dup_share > 0:
                self.df2 = self.df2.drop_duplicates(
                    subset=self.merge_cols,
                    keep=self.keep_dups,
                ).reset_index(drop=True)
        else:
            print('Drop duplicates parameter set to FALSE. Merging with duplicates.')
        
    def check_col_exists(self, col: str) -> List[str]:
        """Check that column exists in df1 and df2."""
        errors_list = []
        if col not in self.df1.columns:
            errors_list.append(f'Merge column "{col}" not in df1.')
        if col not in self.df2.columns:
            errors_list.append(f'Merge column "{col}" not in df2.')
        return errors_list

    def check_common_values(self, col:str) -> List[str]:
        """Check that column has common values in df1 and df2."""
        errors_list = []
        share_df1_col_vals_in_df2 = (self.df1[col].isin(self.df2[col])).mean()
        share_df2_col_vals_in_df1 = (self.df2[col].isin(self.df1[col])).mean()
        print(f'Share of df1 "{col}" values in df2: {round(share_df1_col_vals_in_df2, 2)}')
        print(f'Share of df2 "{col}" values in df1: {round(share_df2_col_vals_in_df1, 2)}')
        if share_df1_col_vals_in_df2 == 0:
            errors_list.append(f'Merge column "{col}" has no common values in df1 and df2.')
        return errors_list

    def check_same_dtype(self, col:str) -> List[str]:
        """Check that column has same dtype in df1 and df2."""
        errors_list = []
        if self.df1[col].dtype != self.df2[col].dtype:
            errors_list.append(f'Merge column "{col}" dtype is not the same in df1 and df2.')
        if self.df1[col].map(type).nunique() > 1:
            errors_list.append(f'Merge column "{col}" in df1 has multiple dtypes in values.')
        if self.df2[col].map(type).nunique() > 1:
            errors_list.append(f'Merge column "{col}" in df2 has multiple dtypes in values.')
        return errors_list
 
## Visualization
def create_gif(
    image_dir: str,
    output_file_dir: str,
    output_file_name: str,
    image_prefix: str = '',
    image_suffix: str = '',
    image_duration_sec: float = 0.5,
) -> None:
    """Create a gif from a group of images located in the same directory."""
    # Add images to list
    images_list = []
    for file in os.listdir(image_dir):
        if file.startswith(image_prefix) and file.endswith(image_suffix):
            images_list.append(imageio.imread(os.path.join(image_dir, file)))
    # Write images to .gif
    imageio.mimwrite(
        uri=os.path.join(output_file_dir, output_file_name),
        ims=images_list,
        duration=image_duration_sec,
    )        
    return None