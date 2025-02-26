




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

csv_path = path / '1_sheets.csv'

csv = pd.read_csv(csv_path)


# csv = csv[['pruning_method', 'alpha', 'retained_flops', 'val IoU', 'val F1', 'test IoU', 'test F1']]



# make the csv have a multiindex that is then easier to work with:

df = csv


# make the csv have a multiindex that is then easier to work with:

df = csv

df.set_index(['retained_flops', 'pruning_method'], inplace=True)

df.index = pd.MultiIndex.from_frame(df.index.to_frame())    # this was for google sheets approach:  .fillna(method='ffill'))  # Fill the NaN values from merged cells in the index
df.sort_index(inplace=True)





# ----- Rename retained_flops to FLOPs, pruning_method to Method
#        make FLOPs into percentage form (like 3% instead of 0.03)
#       rename uniform to Uniform and random to Random 
# -------

# Step 1: Rename the index levels
df = df.rename_axis(index={'retained_flops': 'FLOPs', 'pruning_method': 'Method'})

# Step 2: Convert FLOPs values from decimal to percentage
# Reset the index to access the FLOPs values
df = df.reset_index()

# Convert FLOPs values to percentage format
df['FLOPs'] = df['FLOPs'].apply(lambda x: f"{int(x*100)}%")

# Create a mapping dictionary for the values to rename
# Replace the values in the Method column
rename_mapping = {'uniform': 'Uniform', 'random': 'Random'}
df['Method'] = df['Method'].replace(rename_mapping)

# Set the index back
df = df.set_index(['FLOPs', 'Method', "alpha"])

df.sort_index(inplace=True)








# ----- Make the index be in the correct order -------

def sort_for_sheets(df):

    FLOPs_percents = ["3%", "25%", "50%", "75%", "100%"]
    METHODS = ["L1", "L2", "IPAD_eq", "Uniform", "Random", "no_pruning"]

    df = df.reindex(FLOPs_percents, axis=0, level="FLOPs")
    df = df.reindex(METHODS, axis=0, level="Method")
    return df

df = sort_for_sheets(df)




values = ['val IoU', 'val F1', 'test IoU', 'test F1']

for value in values:

    if value.startswith('val'):
        curr_df = df.loc[pd.IndexSlice[:, ["L1", "L2"]], :]
    else:
        curr_df = df

    result = curr_df.pivot_table(
        values=value,
        index=['FLOPs', 'Method'],
        columns='alpha'
    )
    
    result = sort_for_sheets(result)

    print(result)

    result.to_csv(path / f'sheets_{value}.csv')





# # Drop the no_pruning row
# df_without_no_pruning = df.drop(index=(1.00, 'no_pruning'))

# # If you want to store the no_pruning row separately
# no_pruning_row = df.loc[(1.00, 'no_pruning')]


# df_without_no_pruning.to_csv(path / '2.csv', index=False)
# no_pruning_row.to_csv(path / 'no_pruning.csv', index=False)