from pathlib import Path
import os
from modflowapi.extensions import ApiSimulation
from modflowapi import Callbacks
# from workflow import *
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#add mf6rtm path to the system
sys.path.insert(0,os.path.join("..","..","mf6rtm"))
import flopy
import mf6rtm
import utils

prefix = 'walter1994'
DT_FMT = "%Y-%m-%d %H:%M:%S"

if __name__ == "__main__":
    
    dll = os.path.join(model.wd,"libmf6")
