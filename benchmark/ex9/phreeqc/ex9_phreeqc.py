# python script to run ex9_phreeqc model, preliminary post-processing

# import libraries
import os
import subprocess
import time
import pandas as pd

# fxn to run phreeqc
def run_phreeqc(d, inp_fnm, out_fnm, therm_fnm):

    # Specify the path to phreeqc.exe
    phreeqc_path = os.path.join(d, "phreeqc.exe")  # Replace with the actual relative path to phreeqc.exe

    # Build the command to execute
    command = [
        phreeqc_path,
        os.path.join(d, inp_fnm),
        os.path.join(d, out_fnm),
        os.path.join(d, therm_fnm)
    ]

    try:
        # Use subprocess to run the command
        subprocess.run(command, check=True)
        print("Phreeqc executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error while running Phreeqc: {e}")
        
    # Pause the code for 1 second
    time.sleep(1)

    print("This message is displayed after a 1-second pause.")

# fxn to post process selected output
def prep_simulated_data(d):
    print('prepping simulated data from ex9_phreeqc...')

    # load in simulated data  
    sim_data = pd.read_csv(os.path.join(d,'ex9_phreeqc.sel'), delimiter='\t')
    sim_data.columns = sim_data.columns.str.lstrip()

    # filter to sampling port
    #samp_port = 1 # sampling port 1 @ dist 21 cm = soln 17
    samp_port = 1
    if samp_port == 1:
        sim_data_filt = sim_data[sim_data['soln'] == 17]

    # add minutes
    sim_data_filt['time_mins'] = sim_data_filt['time']/60 # convert from model seconds to minutes
    sim_data_filt = sim_data_filt[sim_data_filt['time_mins'] < 448] # time filter for obs to thin tail a bit
    
    # add obs nme for matching
    sim_data_filt['time_mins'] = sim_data_filt['time_mins'].round(2)
    sim_data_filt = sim_data_filt.drop_duplicates(subset='time_mins')

    # temp patch, check pest
    sim_data_filt = sim_data_filt.sort_values("time_mins").copy()
    sim_data_filt['conc'] = sim_data_filt["EC_uScm_calc"].shift(1)
    sim_data_filt = sim_data_filt.dropna(subset=["conc"]).reset_index(drop=True)

    # add units
    sim_data_filt['conc_unit'] = 'uSpercm'
    
    # save updated sim electrical conductivity as mc_lowpe.sel.prep
    sim_data_red = sim_data_filt[['time_mins', 'conc', 'conc_unit']]
    sim_data_red.to_csv(os.path.join(d,'sim.csv'))

# # set definitions
d = os.getcwd()
inp_fnm = "ex9_phreeqc.txt"
out_fnm = inp_fnm + ".out"
therm_fnm = "llnl.dat"

# # call phreeqc
run_phreeqc(d, inp_fnm, out_fnm, therm_fnm)

# call post-processor
prep_simulated_data(d)
