
You have the original csvs in root folder.
For each of them, make a folder with their name, copy the csv into that folder, name it 0.csv
Then do:

python3 sheets_pypeline/0.py -p [name]
e.g.
python3 sheets_pypeline/0.py -p segnet_150_models_errors_sclera

this created 1_sheets.csv
then do

python3 sheets_pypeline/1_sheets.py -p [name]
e.g.
python3 sheets_pypeline/1_sheets.py -p segnet_150_models_errors_sclera

This creates the other minor csvs. These can then be imported into sheets and moved into the correct places, so you remake the same sheets that were used before.
This way you make it possible to reuse that code.



python3 sheets_pypeline/0.py -p segnet_150_models_errors_sclera
python3 sheets_pypeline/0.py -p segnet_300_models_errors_veins
python3 sheets_pypeline/0.py -p unet_150_models_errors_sclera
python3 sheets_pypeline/0.py -p unet_300_models_errors_veins



python3 sheets_pypeline/1_sheets.py -p segnet_150_models_errors_sclera
python3 sheets_pypeline/1_sheets.py -p segnet_300_models_errors_veins
python3 sheets_pypeline/1_sheets.py -p unet_150_models_errors_sclera
python3 sheets_pypeline/1_sheets.py -p unet_300_models_errors_veins



python3 lightweight_results_3.py F1

Then change in the script to:
CURR_MODE = "IoU"

python3 lightweight_results_3.py IoU