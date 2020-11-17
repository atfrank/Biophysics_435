# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
# mlp for multi-output regression
from annealing import *
from numpy import mean, nan
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
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
