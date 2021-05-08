"""Utility functions."""

# Import packages
import json
import os
from typing import Dict, List, Union

import imageio
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
            
## Memory management
def downcast_df_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    for numeric_type in ['float', 'integer']:
        downcast_cols = list(df.select_dtypes(include=[numeric_type]).columns)
        for col in downcast_cols:
            start_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], downcast=numeric_type)
            end_dtype = df[col].dtype
            if start_dtype != end_dtype:
                print(f'Downcast column {col} from {start_dtype} to {end_dtype}.')
    return df

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