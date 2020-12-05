# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
# mlp for multi-output regression
from annealing import *
from numpy import mean, nan
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from talos.model.normalizers import lr_normalizer
from talos.model import hidden_layers
import talos
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd

def muller_potential_force(x, y, openMM = True):
    """ Defines the 2D Muller potential and returns the potential and force 
        
        x: x-coordinate (float)
        y: y-coordinate (float)
        
        Returns: at position (x,y)
            potential: the potential (float)
            fx: x-component of the force (float)
            fy: y-component of the force (float)            
    """    
    if openMM:
        # https://gist.github.com/rmcgibbo/6094172
        a = np.array([-1, -1, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10, -10, -6.5, 0.7])
        A = np.array([-200, -100, -170, 15])
        x0 = np.array([1, 0, -0.5, -1])
        y0 = np.array([0, 0.5, 1.5, 1])
    else:
        A  = np.array([0,-10,-10,0.5])
        a  = np.array([-1,-1,-6.5,0.7])
        b  = np.array([0,0,11,0.6])
        c  = np.array([-10,-10,-6.5,0.7])
        x0 = np.array([1,0,-0.5,-1])
        y0 = np.array([0,0.5,1.5,1])  
    
    # accumulate potentials
    potential = np.sum(A*np.exp(a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2))
    fx = -np.sum(A*np.exp((a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2))*(2*a*(x-x0)+b*(y-y0)))
    fy = -np.sum(A*np.exp((a*(x-x0)**2+b*(x-x0)*(y-y0)+c*(y-y0)**2))*(2*c*(y-y0)+b*(x-x0)))
    return(potential, fx, fy)

def muller_potential_force_grid(X, Y, openMM = True):
    """ Generates potential and forces from the 2D Muller potential along a 2D mesh grid and returns them as 1D list
        
    X: Arrays with the x-coordinate at grid-point (i,j) (NumPy 2D)
    Y: Arrays with the y-coordinate at grid-point (i,j) (NumPy 2D)
    openMM: Whether to use the openMM definition of the potential (logical: default = True)
    
    Returns 1D arrays (list) containing:
        xs: the x-coordinates in the grid
        ys: the x-coordinates in the grid
        potential: the potentials along the grid
        forcex: the x-component of the forces along the grid
        forcey: the y-component of the force along the grid
    """        
    
    # get shape of the grid and then compute and store results
    nr, nc = X.shape[0], X.shape[1]    
    xs, ys, potential, forcex, forcey = [],[],[],[],[]
    for i in range(nr):
        for j in range(nc):
            x, y = X[i,j], Y[i,j] 
            p, fx, fy = muller_potential_force(x, y, openMM = True)
            xs.append(x)
            ys.append(y)
            potential.append(p)
            forcex.append(fx)
            forcey.append(fy)
    # return the potential grid
    return(xs, ys, potential, forcex, forcey)

def generate_potential_force_data_grid(x_min = -2.0, x_max = 1.0, y_min = -1.0, y_max = 2.5, delta = 0.025):
    """ Sets up a 2D mesh grid and generates potential and forces from the 2D Muller potential along it
    
    x_min: minimum value for the x-coordinate (float: default = -2.0)
    x_max: maximum value for the x-coordinate (float: default = 1.0)
    y_min: minimum value for the y-coordinate (float: default = -2.0)
    y_max: maximum value for the y-coordinate (float: default = 2.5)
    delta: spacing between grid points (float: default = 0.025)
    
    Returns NumPy 2D arrays containing:
        X: x-coordinate at grid-point (i,j)
        Y: y-coordinate at grid-point (i,j)
        potential: potential at grid-point (i,j)
        forcex: x-component of the force at grid-point (i,j)
        forcey: y-component of the force at grid-point (i,j)
    """        
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    X, Y, potential, forcex, forcey = muller_potential_force_grid(X, Y)
    return(X, Y, potential, forcex, forcey)

def muller_potential_grid_ml(model, X, Y):
    """ Uses a model to compute the 2D Muller potential over a grid at points specified in X and Y
    
    model: Keras machine learning model (Keras model)
    X: NumPy 2D arrays containing the x-coordinate at grid-point (i,j)
    Y: NumPy 2D arrays containing the y-coordinate at grid-point (i,j)    
    
    Returns:
        potential: a NumPy 2D array containing potential at grid-point (i,j)
    """    
    # get shape of the grid and then compute and store results
    kb = 0.001985875
    nr, nc = X.shape[0], X.shape[1]    
    potential = np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            x, y = X[i,j], Y[i,j] 
            ene, fx, fy = predict([x, y], model)
            potential[i,j] = ene
    # return the potential grid
    return(potential)

def plot_muller_potential_ml(model, x_min = -2.0, x_max = 1.0, y_min = -1.0, y_max = 2.5, delta = 0.025, origin = 'lower', colors = plt.cm.viridis, clip_max = 120.0, clip_min = -150.0, color_bar = "Energy", title = 'Muller Potential'):
    """ Plots the 2D Muller potential learn by a machine learning model
    
    model: Keras machine learning model (Keras model)
    x_min: minimum value for the x-coordinate (float: default = -2.0)
    x_max: maximum value for the x-coordinate (float: default = 1.0)
    y_min: minimum value for the y-coordinate (float: default = -2.0)
    y_max: maximum value for the y-coordinate (float: default = 2.5)
    delta: spacing between grid points (float: default = 0.025)
    origin: the origin for plotting (string: default = "lower")
    colors: matplotlib color map used for plotting (matplotlib color map: default = plt.cm.viridis)    = plt.cm.viridis
    clip_max: maximum value above which all values are ignored (float: default = 120.0)
    clip_min: minimum value below which all values are ignored (float: default = -150.0)
    color_bar: label for the colorbar (string: default = "Energy")
    title: label for the plot (string: defalut = 'Muller Potential'
    
    Returns NumPy 2D arrays containing the data used for plotting:
        X: x-coordinate at grid-point (i,j)
        Y: y-coordinate at grid-point (i,j)
        Z: potential at grid-point (i,j)    
    """    
    # set up grid and compute potential
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    Z = muller_potential_grid_ml(model, X, Y)
    nr, nc = Z.shape

    # plot contours
    fig1, ax2 = plt.subplots(constrained_layout=True)
    CS = ax2.contourf(X, Y, Z.clip(max=clip_max, min=clip_min), 10, cmap=colors, origin=origin)
    CS2 = ax2.contour(CS, levels=CS.levels[::1], colors='r', origin=origin)
    
    # labels
    ax2.set_title(title)
    ax2.set_xlabel('Coordinate-X')
    ax2.set_ylabel('Coordinate-Y')

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel(color_bar)
    
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)
    
    return(X, Y, Z)

# functions for machine learning

def get_dataset(system = "2D_Muller", x_min = -2.0, x_max = 1.0, y_min = -1.0, y_max = 2.5, delta = 0.025):
    
    """ Used to get dataset used for training a machine learning model
    
    system: indicates whether to return data for the 2D-Muller potential or RNA (string: default = '2D_Muller')
    
    For the Muller potential:
        x_min: minimum value for the x-coordinate (float: default = -2.0)
        x_max: maximum value for the x-coordinate (float: default = 1.0)
        y_min: minimum value for the y-coordinate (float: default = -2.0)
        y_max: maximum value for the y-coordinate (float: default = 2.5)
        delta: spacing between grid points (float: default = 0.025)
    
    Returns:
        targets: a NumPy 2D array the values of the targets that will be used for machine learning (each row is a different sample, i.e., datapoint)
        features: a NumPy 2D array the values of the features that will be used for machine learning (each row is a different sample, i.e., datapoint)
        
    """       
    if system == "2D_Muller":
        X, Y, potential, forcex, forcey = generate_potential_force_data_grid(x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, delta = delta)
        features = pd.DataFrame({"X": X, "Y": Y})
        targets = pd.DataFrame({"potential": potential, "fx": forcex, "fy": forcey})
    if system == "RNA":
        pass    
    return(targets.values, features.values)


def get_model(n_inputs, n_outputs, loss):
    """ Initializes a sequential neural network machine learning model 
    
    The model created has:
        an input layer, which is connected to
        the first layer has 12 neurons, which is connected to
        the second layer has 6 neurons, which is connected to
        output layer (contains the predicted values of the target)
    
    Returns:
        model: a Keras neural network model
    
    """    
    model = Sequential()
    model.add(Dense(25, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer='adam')
    return model
 

# first we have to make sure to input data and params into the function
def talos_model(x_train, y_train, x_val, y_val, params):
    # https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
    # next we can build the model exactly like we would normally do it
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))

    model.add(Dense(y_train.shape[1], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'],
                      metrics=['acc'])
    history = model.fit(x_train, y_train, 
                            validation_data=(x_val, y_val),
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=0)    
    # finally we have to make sure that history object and model are returned
    return history, model

def talos_search_space():
    p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2, 4, 8],
     'batch_size': (2, 30, 10),
     'epochs': [150],
     'dropout': (0, 0.5, 5),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'optimizer': ['Adam', 'Nadam', 'RMSprop'],
     'kernel_initializer': ['uniform','normal'],
     'losses': ['mean_squared_error', 'mean_absolute_error'],
     'activation':['relu', 'elu'],
     'last_activation': ['relu', 'elu']}
    return(p)

def best_talos_parameters():
    p = {'lr': 4.10,
     'first_neuron': 64,
     'hidden_layers': 2,
     'batch_size': 21,
     'epochs': 150,
     'dropout': 0,
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'optimizer': 'Nadam',
     'kernel_initializer': 'uniform',
     'losses': 'mean_absolute_error',
     'activation': 'elu',
     'last_activation': 'elu'}
    return(p)

def train_model(y, X, y_val=None, X_val=None, verbose=0, epochs=1000, batch_size=10, loss = "mae", callbacks = []):
    """ Train a Keras model
    
    y: Numpy array with training targets (this is what we want to predict)
    X: Numpy array with training features (the input that we'll use to make our predictions)
    verbose=0
    epochs: number of training epochs (integer: default = 1000) during each epoch, all the data is run through the networks this is done in batches
    batch_size: the number of samples to use for each training batch (integer: default =10)
    loss: the loss function used to optimize the model (string: default = "mae")
        see: https://keras.io/api/losses/regression_losses/
        see: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
           
    Returns:
        model: an optimized Keras neural network model
    
    """     
    # get dimensions of the features and targets
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    
    # initialize model
    model = get_model(n_inputs, n_outputs, loss)
    
    # fit model
    if X_val is None or y_val is None:
        X_val, yval = X, y
        print("Warning: created fake validation set")
    else:
        print("Warning: Working with user supplied validation set")
        
    model.fit(X, y, verbose=verbose, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=(X_val, y_val))
    
    # return model
    return model
    
def evaluate_model(y, X):
    """ Evaluates a model using cross-validation 
    
    y: Numpy array with training targets (this is what we want to predict)
    X: Numpy array with training features (the input that we'll use to make our predictions)
           
    Returns:
        results: list with the cross-validation statistics
        model: a Keras neural network model
    
    """     
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=1000)
        # evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=1)
        # store result
        print('>%.3f' % mae)
        results.append(mae)        
    return results, model


def predict(state, model, system = "2D_Muller"):
    """ Helper function to make prediction using the trained model 
    
        state: instance of your system: for 2D Muller potential this is the list of the x- and y-coordinates
        model: machine learning model
        system: indicates whether to return predictions for the 2D-Muller potential or RNA (string: default = '2D_Muller')
        
    """    
    if system == "2D_Muller":
        result = model.predict(np.array([state]))
        return(result[0][0], result[0][1], result[0][2])
    if system == "RNA":
        pass


def get_resname_int(resname):
    if resname == 'A' or resname == 'ADE':
        return 1
    elif resname == 'G' or resname == 'GUA':
        return 2
    elif resname == 'C' or resname == 'CYT':
        return 3
    elif resname == 'U' or resname == 'URA':
        return 4

def get_resname_char(resname):
    if resname == 1:
        return 'ADE'
    elif resname == 2:
        return 'GUA'
    elif resname == 3:
        return 'CYT'
    elif resname == 4:
        return 'URA'

def cT2features(ct, rna, is_dataframe = False):
    # function to extract features from DSSR annotated secondary structure (.ct)
    if is_dataframe:
        df = ct
    else:
        df = pd.read_csv(ct, delim_whitespace=True,header=None,skiprows=1)
         
    length = len(df)
    df.columns = ['resid','resname','i_minus_1','i_plus_1','i_bp','resid2']

    #residue name of residue i
    i_resname = df['resname'].apply(lambda x: get_resname_int(x)) # Series
    i_resname_char = df['resname'].apply(lambda x: get_resname_char(get_resname_int(x))) 

    #residue that base paired to i is (resname instead of resid):
    i_bp = df['i_bp']
    i_bp_resname = []
    for i in range(length):
      if i_bp[i] == 0:
        i_bp_resname.append(0)
      else:
        i_bp_resname.append(get_resname_int(df[df['resid']==i_bp[i]]['resname'].values[0]))
    i_bp_resname = pd.Series(i_bp_resname)

    #residue that base paired to i-1 is:
    i_minus_1_bp = df['i_bp'].values.tolist()
    i_minus_1_bp = [0]+i_minus_1_bp # add 0 at the begining
    i_minus_1_bp.pop() # delete last element
    i_minus_1_bp_resname = []
    for i in range(length):
      if i_minus_1_bp[i] == 0:
        i_minus_1_bp_resname.append(0)
      else:
        i_minus_1_bp_resname.append(get_resname_int(df[df['resid']==i_minus_1_bp[i]]['resname'].values[0]))
    i_minus_1_bp_resname = pd.Series(i_minus_1_bp_resname)

    #residue that base paired to i+1 is:
    i_plus_1_bp = df['i_bp'].values.tolist()
    i_plus_1_bp = i_plus_1_bp+[0] # add 0 at the end
    i_plus_1_bp.pop(0) # delete last element
    i_plus_1_bp_resname = []
    for i in range(length):
      if i_plus_1_bp[i] == 0:
        i_plus_1_bp_resname.append(0)
      else:
        i_plus_1_bp_resname.append(get_resname_int(df[df['resid']==i_plus_1_bp[i]]['resname'].values[0]))
    i_plus_1_bp_resname = pd.Series(i_plus_1_bp_resname)

    #residue name of i-1
    i_minus_1_resname = i_resname.values.tolist()
    i_minus_1_resname = [0] + i_minus_1_resname
    i_minus_1_resname.pop()
    i_minus_1_resname = pd.Series(i_minus_1_resname)

    #residue name of i+1
    i_plus_1_resname = i_resname.values.tolist()
    i_plus_1_resname = i_plus_1_resname+[0]
    i_plus_1_resname.pop(0)
    i_plus_1_resname = pd.Series(i_plus_1_resname) 

    #the previous residue of the residue base paired to i: i_bp_minus_1
    i_bp_minus_1 = list(map(lambda x: x-1, i_bp.values.tolist()))
    i_bp_minus_1_resname = []
    for i in range(length):
      if i_bp_minus_1[i]==-1 or i_bp_minus_1[i]==0:
        i_bp_minus_1_resname.append(0)
      else:
        i_bp_minus_1_resname.append(get_resname_int(df[df['resid']==i_bp_minus_1[i]]['resname'].values[0]))
    i_bp_minus_1_resname = pd.Series(i_bp_minus_1_resname)

    #the next residue of the residue base paired to i: i_bp_plus_1
    i_bp_plus_1 = list(map(lambda x: x+1, i_bp.values.tolist()))
    i_bp_plus_1_resname = []
    for i in range(length):
      if i_bp_plus_1[i]==1 or i_bp_plus_1[i]==(length+1):
        i_bp_plus_1_resname.append(0)
      else:
        i_bp_plus_1_resname.append(get_resname_int(df[df['resid']==i_bp_plus_1[i]]['resname'].values[0]))  
    i_bp_plus_1_resname = pd.Series(i_bp_plus_1_resname)

    #i_bp_prev_bp
    i_bp_minus_1_bp_resname = []
    for i in range(length):
      if i_bp_minus_1[i]==-1 or i_bp_minus_1[i]==0:
        i_bp_minus_1_bp_resname.append(0)
      else:
        i_bp_minus_1_bp = df[df['resid']==i_bp_minus_1[i]]['i_bp'].values[0]
        if i_bp_minus_1_bp == 0:
          i_bp_minus_1_bp_resname.append(0)
        else:
          i_bp_minus_1_bp_resname.append(get_resname_int(df[df['resid']==i_bp_minus_1_bp]['resname'].values[0]))
    i_bp_minus_1_bp_resname = pd.Series(i_bp_minus_1_bp_resname)

    #i_bp_next_bp
    i_bp_plus_1_bp_resname = []
    for i in range(length):
      if i_bp_plus_1[i]==1 or i_bp_plus_1[i]==(length+1):
        i_bp_plus_1_bp_resname.append(0)
      else:
        i_bp_plus_1_bp = df[df['resid']==i_bp_plus_1[i]]['i_bp'].values[0]
        if i_bp_plus_1_bp == 0:
          i_bp_plus_1_bp_resname.append(0)
        else:
          i_bp_plus_1_bp_resname.append(get_resname_int(df[df['resid']==i_bp_plus_1_bp]['resname'].values[0]))  
    i_bp_plus_1_bp_resname = pd.Series(i_bp_plus_1_bp_resname) 

    # create other information as Series
    rnaid = pd.Series([rna]*length)
    total_length = pd.Series([length]*length)
    resid = df['resid']
    resname = df['resname']

    #print(features)
    features = pd.concat([rnaid, total_length, resid, resname, i_resname, i_minus_1_resname,
    i_plus_1_resname, i_bp_resname, i_minus_1_bp_resname, i_plus_1_bp_resname, i_bp_minus_1_resname,
    i_bp_plus_1_resname, i_bp_minus_1_bp_resname, i_bp_plus_1_bp_resname], axis=1)
    features.columns = ["id","length","resid","i_resname_char","i_resname","i_minus_1_resname","i_plus_1_resname",
                        "i_bp_resname", "i_minus_1_bp_resname", "i_plus_1_bp_resname", "i_bp_minus_1_resname",
                        "i_bp_plus_1_resname", "i_bp_minus_1_bp_resname", "i_bp_plus_1_bp_resname"]
    return features

def rename_nucleus_type(nucleus):
  if "'" in nucleus:
    return nucleus.replace("'","p")
  else:
    return nucleus

def load_data(id, type = "2D"):
    """ load chemical shifts and get features for a specific dataset """ 
    cs = pd.read_csv("data/chemical_shifts/%s.csv"%id, header=0, sep = "\s+")
    
    if type == "2D":
        features = cT2features("data/secondary_structures/%s.ct"%id, id)
    elif type == "3D":
        features
    else:
        raise AssertionError()
    tmp = cs.merge(features, on = ['id', 'resid'])
    return(tmp[cs.columns], tmp[features.columns])

def fit_hotencoder(all_features):
    """ execute one-hot encoding of the features """
    X = all_features.drop(['id','length','resid'], axis=1)
    enc = preprocessing.OneHotEncoder(sparse = False)
    enc.fit(X)
    return(enc.transform(X), enc)

def scaler(data):
    """ applying standard scaler to data """
    scaler = StandardScaler()
    scaler.fit(data)
    scaler.transform(data)
    return (scaler)

def load_entire_database(type = "2D", split_database = 0.2, id_list_file = "data/chemical_shifts/id.txt", header = 0):
    """ loads chemical shifts and features for the entire  dataset """     
    # read in file with list of ids (here simply the PDB ID associated with each RNA)
    ids = pd.read_csv(id_list_file, header=header) 
    
    # initialize list that will store chemical shifts and features
    all_cs, all_features = [],[]
    
    # loop over ids and get/generate data
    for id in ids.id:
        cs, features = load_data(id, type)    
        all_cs.append(cs)
        all_features.append(features)
    all_cs = pd.concat(all_cs)    
    all_features = pd.concat(all_features)    
    
    # carry out one-hot encoding of features to be used for training model
    X, enc = fit_hotencoder(all_features)    
    
    # store entire database in a dictionary and return to user
    database = {}
    database['one-hot-encoder'] = enc
    database['raw_features'] = all_features
    database['raw_targets'] = all_cs
    database['targets'] = all_cs.drop(['id', 'resid', 'resname', 'class'], axis=1)
    database['features'] = X
    
    # split data
    if split_database > 0: database['features_train'], database['features_test'], database['targets_train'], database['targets_test'] = train_test_split(database['features'], database['targets'], test_size = 0.2, random_state=42)
        
    # scale targets and add to database
    database['scaler'] = scaler(database['targets_train'])
    database['targets_train_scaled'] = database['scaler'].transform(database['targets_train'])
    database['targets_test_scaled'] = database['scaler'].transform(database['targets_test'])
    return(database)

def CS_list_merge(expCS, predCS, nuclei):
    df = []
    for i,nucleus in enumerate(nuclei):
        v1 = expCS.iloc[:,i].values
        v2 = predCS.iloc[:,i].values
        n = [nucleus for i in range(len(v1))]
        df.append(pd.DataFrame({"nucleus": n, "expCS": v1, "predCS": v2}))
    df = pd.concat(df)
    return(df)


def setup_tensorboard(model_path = './new_logs'):    
    try:
        os.rmdir(model_path) 
    except:
        pass

    tensorboard = TensorBoard(
      log_dir=model_path,
      histogram_freq=1,
      write_images=True
    )
    return(tensorboard)

    
def get_model_uncertainity(model, database, output_file):
    """ Compute the uncertainity of a model 
        Input:
            model: trained Keras model
            database: dictionary storing the data for training and testing the model
            output_file: path to output file that will store the information
        Returns:
            uncertainity: dataFrame with uncertainity estimates for a model    
            chemical_shifts: dataFrame with predicted and experimental chemical shfit paired
    """
    nuclei = "C1' C2' C3' C4' C5' C2 C5 C6 C8 H1' H2' H3' H4' H5' H5'' H2 H5 H6 H8".split()
    
    # create DataFrame with (a) predicted and then (b) experimental chemical shifts
    predCS = pd.DataFrame.from_records(database['scaler'].inverse_transform(model.predict(database['features_test'])))
    expCS =  pd.DataFrame.from_records(database['scaler'].inverse_transform(database['targets_test_scaled']))
    
    # merge them
    chemical_shifts = CS_list_merge(expCS, predCS, nuclei)
    chemical_shifts['error'] = np.absolute(chemical_shifts.expCS.values - chemical_shifts.predCS.values)
    initial = predCS.columns.values
    predCS.columns = ["pred_"+str(i) for i in initial]
    expCS.columns = ["exp_"+str(i) for i in initial]
    
    # compute errors
    maes = []
    for i,nucleus in enumerate(nuclei):
        tmp = chemical_shifts[chemical_shifts.nucleus==nucleus]
        mae = mean_absolute_error(tmp.expCS.values, tmp.predCS.values)
        maes.append(mae)
        print("Nucleus: %s MAE: %4.3f ppm"%(nucleus, mae))
    uncertainity = pd.DataFrame({"nucleus": nuclei, "error":maes})
    uncertainity.to_csv(output_file, sep=' ', header=None, index=False)
    return(uncertainity, chemical_shifts)


def features2CS(features, model, encoder, scaler):
    """ Computes chemical shifts from a set of features
        Input:
            features: raw features returned by ```cT2features()```
            model: trained Keras model
            one-hot encoder: feature encoder (e.g. one stored in the database dictionary returned by ```load_entire_database()```)
            scaler: target scaler (e.g. one stored in the database dictionary returned by ```load_entire_database()```)      
        Returns:
            predCS: dataFrame with predicted (computed) chemical shifts
    """
    nuclei = "C1' C2' C3' C4' C5' C2 C5 C6 C8 H1' H2' H3' H4' H5' H5'' H2 H5 H6 H8".split()    
    X = features.drop(['id','length','resid'], axis=1)
    X = encoder.transform(X)    
    predCS = pd.DataFrame.from_records(scaler.inverse_transform(model.predict(X)))    
    predCS.columns = nuclei
    return(predCS)