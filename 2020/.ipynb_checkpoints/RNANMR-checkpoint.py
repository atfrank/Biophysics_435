import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard

DIR_PATH = '/content/drive/My Drive/RESEARCH/ss2cs/'

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

def cT2features(ct_path, rna):
    # function to extract features from DSSR annotated secondary structure (.ct)
    df = pd.read_csv(ct_path, delim_whitespace=True,header=None,skiprows=1)
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
    for i,nucleus in enumerate(nuclei.values):
        v1 = expCS.iloc[:,i].values
        v2 = predCS.iloc[:,i].values
        n = [nucleus[0] for i in range(len(v1))]
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
