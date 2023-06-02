#! /usr/bin/env python3

import os

file_path = '/Users/pn22327/Documents/foulnessdata/code/JP52_modelling/4param_tempdf.csv'

os.system(f'python /Users/pn22327/Documents/foulnessdata/code/JP52_modelling/heatmaps_gvecminplus1p25hzmin.py "{file_path}"')

os.system(f'python /Users/pn22327/Documents/foulnessdata/code/JP52_modelling/heatmaps_gvecMin.py "{file_path}"')

os.system(f'python /Users/pn22327/Documents/foulnessdata/code/JP52_modelling/heatmap_branch3Freq.py "{file_path}"')

os.system(f'python /Users/pn22327/Documents/foulnessdata/code/JP52_modelling/heatmaps_gvecMinFreq.py "{file_path}"')