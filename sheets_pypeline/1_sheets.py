

import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024)
handlers = py_log_always_on.file_handler_setup(MY_LOGGER)






import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
# arg path -p
parser.add_argument('-p', '--path', type=str, help='path to the data folder. e.g. segnet_150_models_errors_sclera   ')

args = parser.parse_args()
path = Path(args.path)

csv_path = path / '1.csv'

csv = pd.read_csv(csv_path)
py_log_always_on.log_manual(MY_LOGGER, csv=csv)


# csv = csv[['pruning_method', 'alpha', 'retained_flops', 'val IoU', 'val F1', 'test IoU', 'test F1']]



# make the csv have a multiindex that is then easier to work with:

df = csv



print(df.head())


result = df.pivot_table(
    values='test IoU',
    index=['pruning_method', 'retained_flops'],
    columns='alpha'
)

print(result)