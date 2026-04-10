import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from peket.kn_side.utils import plot_param_evolution

'''
Example script to show how to plot the results of the ts loop tutorial for 1 grid.
'''

DIR = f"peket/example_file/ts-loop-tutorial/bu19_opt"  # change as needed
# control shape of plot
col_num = 5
row_num = 5 

UL = False # if UL are present in the data (not the case for the example data)

true_merger = '2020-01-07T00:00:00.000' # change as needed (keep the same for all the analyses to see how the timeshift evolves)

minus_num = 9 # max number of removed points + 1 (to include the full data analysis as well)

MODEL = 'Bu2019lm' # change as needed (Bu2019lm, Bu2026_MLP, Ka2017)

plot_param_evolution(DIR=DIR, MODEL=MODEL, col_num=col_num, row_num=row_num, minus_num=minus_num, true_merger=true_merger, UL=UL)
