from mf6rtm import mf6rtm
import os

__author__ = "Pablo Ortega"

def run_cmd():
    # get the current directory
    cwd = os.getcwd()
    # run the solve function
    mf6rtm.solve(cwd)
