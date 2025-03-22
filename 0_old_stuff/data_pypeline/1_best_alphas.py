

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

df.set_index(['pruning_method', 'alpha', 'retained_flops'], inplace=True)

df.index = pd.MultiIndex.from_frame(df.index.to_frame())    # this was for google sheets approach:  .fillna(method='ffill'))  # Fill the NaN values from merged cells in the index
df.sort_index(inplace=True)



# create empty df
columns = ['pruning_method', 'best_alpha', 'best_value']
df_iou = pd.DataFrame(columns=columns)
df_f1 = pd.DataFrame(columns=columns)


# Example of how to use the multiindex:

for pr_method in df.index.unique(level='pruning_method'):

    if pr_method == "no_pruning":
        continue


    df_0 = df.loc[pr_method]
    # cross selecting for retained_flops = 0.25
    df_0 = df_0.xs(0.25, level='retained_flops')


    df_0_iou = df_0['val IoU']
    df_0_f1 = df_0['val F1']

    # get best iou in this df
    best_iou = df_0_iou.max()
    best_alpha_iou = df_0_iou.idxmax()

    best_f1 = df_0_f1.max()
    best_alpha_f1 = df_0_f1.idxmax()

    new_row_iou = pd.DataFrame({'pruning_method': [pr_method], 'best_alpha': [best_alpha_iou], 'best_value': [best_iou]})
    new_row_f1 = pd.DataFrame({'pruning_method': [pr_method], 'best_alpha': [best_alpha_f1], 'best_value': [best_f1]})

    df_iou = pd.concat([df_iou, new_row_iou], ignore_index=True)
    df_f1 = pd.concat([df_f1, new_row_f1], ignore_index=True)




to_iou_csv = path / 'best_0.25_IoU.csv'
to_f1_csv = path / 'best_0.25_F1.csv'

df_iou.to_csv(to_iou_csv)
df_f1.to_csv(to_f1_csv)


sys.exit(0)


