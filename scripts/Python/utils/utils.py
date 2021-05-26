"""Utility functions."""

# Import packages
from collections import OrderedDict
import json
import os
from typing import Dict, List, Union, Optional

import imageio
import numpy as np
import pandas as pd

## I/O
def load_json(filepath:str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def create_dirs(dirs: Union[str, List[str]]) -> None:
    """Create single/multiple directories.

    Arguments
        dirs: directory path(s)
    """
    dirs = [dirs] if not isinstance(dirs, list) else dirs
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f'Directory "{path}" already exists.')

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

## Memory management
def check_memory_usage(df: pd.DataFrame) -> float:
    """Check memory usage (in MB) for dataframe object."""
    return df.memory_usage(index=False, deep=True).sum() / (1024**2)

def downcast_df_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
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