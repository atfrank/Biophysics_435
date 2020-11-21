get_ipython().run_cell_magic("capture", "", """from RNANMR import load_entire_database, CS_list_merge,setup_tensorboard
from machine_learning import train_model, best_talos_parameters
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error
import os""")


# inspect this database dictionary
database = load_entire_database()
# training targets: database['targets_train_scaled']
# testing targets:  database['targets_test_scaled']
# training features: database['features_train']
# testing features:  database['features_test']


history, model = #add code here


import joblib
filename = 
loaded_model = joblib.load(filename)


# add code here


# add code here


# add code here


# add code here
