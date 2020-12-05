get_ipython().run_cell_magic("capture", "", """# Plotly and Talos
#!pip install plotly==4.12.0
#!pip install jupyterlab "ipywidgets>=7.5"
#!jupyter labextension install jupyterlab-plotly@4.12.0
#!pip install talos --use-feature=2020-resolver""")


get_ipython().run_cell_magic("capture", "", """from machine_learning import *
import pandas as pd
import plotly.express as px
import numpy as np
import os""")


database = load_entire_database()


# Define Tensorboard as a Keras callback
keras_callbacks = [
  setup_tensorboard(model_path = './new_logs')
]

model = train_model(y=database['targets_train_scaled'], 
                    X=database['features_train'],
                    y_val = database['targets_test_scaled'],
                    X_val = database['features_test'],
                    verbose=1,
                    epochs=50, 
                    batch_size=1000, 
                    loss='mean_squared_error', callbacks=keras_callbacks)


unc, cs = get_model_uncertainity(model, database, output_file = "test.txt")


# single feature instance
feature = database['raw_features'].iloc[0].to_frame().transpose()
# the hot-one-encoder
encoder = database['one-hot-encoder']
# scaler
scaler = database['scaler']
# call function
features2CS(feature, model, encoder, scaler)


fig = px.scatter(cs, x="expCS", y="predCS", color="nucleus", width = 400, height = 400)
fig.show()


fig = px.violin(cs, y="error", color="nucleus", box=True, points="all",
          hover_data=cs.columns)
fig.show()
