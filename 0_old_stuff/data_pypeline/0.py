

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

parser = argparse.ArgumentParser()
# arg path -p
parser.add_argument('-p', '--path', type=str, help='path to the data folder. e.g. segnet_150_models_errors_sclera   ')

args = parser.parse_args()
path = Path(args.path)

csv_path = path / '0.csv'

csv = pd.read_csv(csv_path)
py_log_always_on.log_manual(MY_LOGGER, csv=csv)


# take only columns: model, test F1, test IoU, val F1, val IoU
csv = csv[['model', 'test F1', 'test IoU', 'val F1', 'val IoU']]
py_log_always_on.log_manual(MY_LOGGER, csv=csv)


# split model column into pruning_method column and retained_flops_percent column
def pruning_method(x):
    base = x.split('_')[:-1]
    concated = '_'.join(base)
    concated = 'no_pruning' if concated == '' else concated
    return concated
csv['pruning_method'] = csv['model'].apply(pruning_method)
csv['retained_flops_percent'] = csv['model'].apply(lambda x: x.split('_')[-1])
py_log_always_on.log_manual(MY_LOGGER, csv=csv)


# get retained flops as float
convert_percent = lambda x: float(x.rstrip('%')) / 100 if isinstance(x, str) else float(x)
csv['retained_flops'] = csv['retained_flops_percent'].apply(convert_percent)
py_log_always_on.log_manual(MY_LOGGER, csv=csv)





# get the transformation into alphas and methods

alphas = {
    "IPAD1_L1": 0.5,
    "IPAD2_L2": 0.5,
    "IPAD1": 0.0,
    "IPAD2": 0.0,
    "L1": 1.0,
    "L2": 1.0,
}

methods = {
    "IPAD1_L1": "L1",
    "IPAD2_L2": "L2",
    "IPAD1": "L1",
    "IPAD2": "L2",
    "L1": "L1",
    "L2": "L2",
}

def get_alpha(x):
    if not x in alphas:
        return None
    return alphas[x]

def get_method(x):
    if not x in methods:
        return x
    return methods[x]

csv['alpha'] = csv['pruning_method'].apply(get_alpha)
csv['pruning_method'] = csv['pruning_method'].apply(get_method)
py_log_always_on.log_manual(MY_LOGGER, csv=csv)



# now make it only be these columns:   pruning_method alpha retained_flops test_IoU test_F1 retained_flops_percent
csv = csv[['pruning_method', 'alpha', 'retained_flops', 'val IoU', 'val F1', 'test IoU', 'test F1']]
py_log_always_on.log_manual(MY_LOGGER, csv=csv)






# make the csv have a multiindex that is then easier to work with:

df = csv

df.set_index(['pruning_method', 'alpha', 'retained_flops'], inplace=True)

df.index = pd.MultiIndex.from_frame(df.index.to_frame())    # this was for google sheets approach:  .fillna(method='ffill'))  # Fill the NaN values from merged cells in the index
df.sort_index(inplace=True)




# Example of how to use the multiindex:

for pr_method in df.index.unique(level='pruning_method'):
    df_0 = df.loc[pr_method]

    # print(df_0)
    print(pr_method)
    
    for alpha in df_0.index.unique(level='alpha'):
        df_1 = df_0.loc[alpha]

        for retained_flops in df_1.index.unique(level='retained_flops'):
            df_2 = df_1.loc[retained_flops]
            # if pr_method not in {"L1", "L2"}:
            #     alpha = None

            # if pr_method != "no_pruning" and retained_flops == 1.0:
            #     continue

            # if pr_method == "no_pruning" and retained_flops != 1.0:
            #     continue

            print(df_2['test IoU'])

print(df.index.unique(level='pruning_method'))

to_csv = path / '1.csv'
df.to_csv(to_csv)

# # this proves that multiindex isn't sth that would be obvious from a file
# df = pd.read_csv("test.csv")
# print(df)
# print(df.index)


