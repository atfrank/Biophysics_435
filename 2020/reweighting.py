import pandas as pd
import numpy as np
import os
import BME.bme_reweight as bme

def BME_find_weights(expcs, sim, theta, conformers):
  # generate random weights 
  w0 = np.random.random(conformers)
  # normalize
  w0 /= np.sum(w0)

  # initialize reweighting class with weights                                                                                                                                
  bmea = bme.Reweight(verbose=False, w0=list(w0))
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return list(np.round_(bmea.get_weights(), 5)), list(np.round_(chi2_after, 5))
 
def save_bme_results(tmp, filename):
  mean_w, sd_w = [], []
  for i in range(0, tmp.shape[0]):
    w = []
    tmp2 = tmp[i,:,:]    
    for j in range(0, tmp2.shape[1]):
      w.append(np.mean(tmp2[:,j]))
      w.append(np.std(tmp2[:,j]))
    mean_w.append(w)
  pd.DataFrame.from_records(mean_w).to_csv(filename, sep=' ', header=None, index=False)

def add_accuracy(row):
  if "C" in row['nucleus']:
    return 0.84
  else:
    return 0.11

def BME_find_theta(expcs, sim, theta):
  bmea = bme.Reweight(verbose=False)
  # load data
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return chi2_after

  # initialize reweighting class with weights                                                                                                                                
  bmea = bme.Reweight(verbose=False)
  bmea.load(expcs, sim)
  chi2_before,chi2_after, srel = bmea.optimize(theta=theta)
  return list(np.round_(bmea.get_weights(), 5))

def null_weights(N=18):
  return([float(1/N) for i in range(0, N)])

def load_ss2cs_model(nucleus, rna = '', DIR_PATH = '/content/drive/My Drive/RESEARCH/ss2cs/'):
  ''' load save model '''
  filename = DIR_PATH + 'output/RF_'+ rna + nucleus + '.sav'
  model = pickle.load(open(filename, 'rb'))
  return(model)


def write_BME_chemical_shifts(input_exp = "data/chemical_shifts/measured_shifts_ 7JU1.dat", 
                              input_sim = "data/chemical_shifts/computed_shifts_7JU1.dat",
                              input_accuracy = "data/chemical_shifts/uncertainity.dat",
                              output_name_exp = "data/chemical_shifts/bme_experimental_7JU1.dat", 
                              output_name_sim = "data/chemical_shifts/bme_simulated_7JU1.dat"):
    """ Writes out files chemical shift files needed for BME 
            input_exp: path to input experimental chemical shift file
            input_sim: path to input computed (simulated) chemical shift file
            input_accuracy: path to input accuracy (uncertainty) file
            output_name_exp: path to output experimental chemical shift file formatted for BME analysis
            output_name_sim: path to output simulated chemical shift file formatted for BME analysis
    """
    # read in experimental chemical shift file
    names = ['resname', 'resid', 'nucleus', 'expCS', 'NA']
    expcs = pd.read_csv(input_exp, sep = "\s+", header = None, names = names)

    # read in computed (simulated) chemical shift file
    names = ['model', 'resid', 'resname', 'nucleus', 'simcs', 'id']
    simcs = pd.read_csv(input_sim, sep = "\s+", header = None, names = names)

    # read in accuracy file
    names = ['nucleus', 'error']
    accuracy = pd.read_csv(input_accuracy, sep = "\s+", header = None, names = names)

    # merge predicted, measured, and accuracy information
    cs = simcs.merge(expcs, on = ['resid', 'resname', 'nucleus']).merge(accuracy, on = ['nucleus'])
    cs = cs.sort_values(by = ['model', 'resid', 'nucleus'])

    # output files for BME
    # experimental
    expcs = cs[['model', 'expCS', 'error']]
    expcs = expcs[expcs.model==1]
    expcs = expcs[['expCS', 'error']]
    expcs.columns = ['DATA=JCOUPLINGS', 'PRIOR=GAUSS']
    expcs.index = [i for i in range(0, expcs.shape[0])]
    expcs.to_csv(output_name_exp, sep = " ", index = True, index_label = "#")
    
    # computed (simulated)
    conformers = cs.model.unique()
    simcs = np.zeros((len(conformers), expcs.shape[0]))
    for j,conformer in enumerate(cs.model.unique()):
        simcs[j,:] = cs[cs.model==conformer].simcs.values        
    simcs = pd.DataFrame.from_records(simcs)
    simcs.to_csv(output_name_sim, sep = " ", index=True, header=None)
    return(expcs, simcs)


def find_weights(exp_file, sim_file, theta):
    """ Find weights using BME 
            exp_file: path to experimental observable file formatted for BME analysis
            sim_file: path to simulated observable file formatted for BME analysis
        Returns:
            weights: optimal weights
            chi2_before: chi^2 before reweighting
            chi2_after: chi^2 after reweighting
            srel: the relative entropy of these optimized weights relative to the initial (prior) weights
    """    
    bmea = bme.Reweight(verbose=True)
    bmea.load(exp_file, sim_file)
    chi2_before, chi2_after, srel = bmea.optimize(theta=theta)
    return bmea.get_weights(), chi2_before, chi2_after, srel
