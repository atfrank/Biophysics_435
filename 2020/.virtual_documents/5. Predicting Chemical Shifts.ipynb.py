get_ipython().run_cell_magic("capture", "", """from RNANMR import load_entire_database, CS_list_merge,setup_tensorboard
from machine_learning import train_model
from keras import backend as K
from keras.callbacks import TensorBoard
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error
import os""")


database = load_entire_database()


database['targets_train_scaled']


# Define Tensorboard as a Keras callback
keras_callbacks = [
  setup_tensorboard(model_path = './new_logs')
]


model = train_model(y=database['targets_train_scaled'], 
                    X=database['features_train'],
                    y_val = database['targets_test_scaled'],
                    X_val = database['features_test'],
                    verbose=1,
                    epochs=500, 
                    batch_size=1000, 
                    loss='mean_squared_error', callbacks=keras_callbacks)


nuclei = pd.read_csv("data/nuclei.txt", header=None, names = ['nucleus'])
predCS = pd.DataFrame.from_records(database['scaler'].inverse_transform(model.predict(database['features_test'])))
expCS =  pd.DataFrame.from_records(database['scaler'].inverse_transform(database['targets_test_scaled']))
#predCS = pd.DataFrame.from_records(model.predict(database['features_test']))
#expCS =  pd.DataFrame.from_records(database['targets_test'])
initial = predCS.columns.values
predCS.columns = ["pred_"+str(i) for i in initial]
expCS.columns = ["exp_"+str(i) for i in initial]


df = CS_list_merge(expCS, predCS, nuclei)
df['error'] = np.absolute(df.expCS.values - df.predCS.values)


fig = px.scatter(df, x="expCS", y="predCS", color="nucleus")
fig.show()


fig = px.violin(df, y="error", color="nucleus", box=True, points="all",
          hover_data=df.columns)
fig.show()


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
for i,nucleus in enumerate(nuclei.values):
    tmp = df[df.nucleus==nucleus[0]]
    mae = mean_absolute_error(tmp.expCS.values, tmp.predCS.values)
    print("Nucleus: get_ipython().run_line_magic("s", " MAE: %4.3f ppm\"%(nucleus[0], mae))")


# parameter optimization: https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
