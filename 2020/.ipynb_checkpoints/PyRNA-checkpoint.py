import numpy as np
import random
from itertools import groupby
from operator import itemgetter
import copy
import matplotlib.pyplot as plt
import network_line_graph as nlg
import pandas as pd

def visualize_structure(bp_matrix, label = 'UUCG-tetraloop', edge_cmap = plt.cm.viridis):
  # initialise figure
  fig, ax = plt.subplots(1,1)

  bp_matrix[bp_matrix == 0] = np.nan
  bp_matrix[np.tril_indices_from(bp_matrix)] = np.nan

  # plot connections above x-axis
  nlg.draw(bp_matrix, arc_above=True, ax=ax)

  ax.text(0,1, label, transform=ax.transAxes, fontsize=18)
  plt.show()
    
def CT2basepair_matrix(CT_file):
    """ Generates base-paired matrix from a CT file """
    col_names = ['n', 'name', 'n-1', 'n+1', 'base-pairing', 'natural_number']
    df = pd.read_csv(CT_file, header = None, sep = " ", skiprows=1,  names=col_names)
    nres = df.shape[0]
    my_shape = (nres, nres)
    bp_matrix = np.zeros(my_shape)

    # populating the base-pair matrix with information for the CT file
    my_cols = (0, 4) # tuple
    bps = df.values[:, my_cols]

    # Loop over bps array and then assign value of 1 on pair that are base-paired
    for bp in bps:
        i,j = bp[0],bp[1]
        if j != 0:
            bp_matrix[i-1,j-1] = 1
    return(bp_matrix, df)

    
def stem2basepair_matrix(sequence, assembled_stems, stems_s1, stems_s2):
    """ Generates base-paired matrix from assembled stems """
    # initialize needed variable
    n = len(sequence)
    bp_i, bp_j = [], []
    # unfold base-pairs  
    for stem in assembled_stems:
        bp_i += stems_s1[stem]
        bp_j += stems_s2[stem]
    
    # generate matrix
    bp_matrix = np.zeros((n, n))
    for i,j in zip(bp_i, bp_j):
        bp_matrix[i, j], bp_matrix[j, i] = 1.0, 1.0
    return(bp_matrix)
    
def random_select_i(items):
    """ Make a random selection from a list 
        Used to randomly select a stem to add or remove from an assembled structure
    """
    return(random.choice(items))

def generate_basepair_matrix(sequence):
    """ Creates the initial matrix that stores base-pairing information """
    n = len(sequence)
    return(np.zeros((n, n)))

def generate_basepair_compatibility_matrix(sequence, bp_types = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']):
    """ Creates:
        1) C_matrix: matrix that specificities if two residues in the RNA can be base-pair 
        if i and j are compable, C_matrix[i,j] and C_matrix[j,i] = 1 else C_matrix[i,j] and C_matrix[j,i] = 0
        2) HB_matrix: the corresponding H-bond counts 
    """
    n = len(sequence)
    purines = ['AU', 'UA']
    C_matrix = np.zeros((n, n))
    HB_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i <= j and j-i > 2:
                res = sequence[i]+sequence[j]
                if res in bp_types: 
                    C_matrix[i, j], C_matrix[j, i] = 1.0, 1.0
                    if res in ['AU', 'UA']:
                        N_HB = 2.0
                    elif res in ['GU', 'UG']:
                        N_HB = 1.5
                    else:
                        N_HB = 3.0
                    G = N_HB
                    HB_matrix[i, j], HB_matrix[j, i] = G, G
    return(C_matrix, HB_matrix)

def setup_stems(bp_compatiable_matrix):
    """ Identifies stems and computes there compatibilities 
        Stems are compatible if they do not cross each other
    """
    stems_s1, stems_s2 = get_stems(bp_compatiable_matrix)
    stem_compatibility_matrix = np.zeros((len(stems_s1), len(stems_s2)))
    stem_indices = []
    for k in range(0, len(stems_s1)):
        stem_k = stems_s1[k] + stems_s2[k]
        stem_indices.append(stem_k)
        for l in range(0, len(stems_s2)):
            stem_l = stems_s1[l] + stems_s2[l]
            if k < l and check_for_overlap(stem_k, stem_l):
                stem_compatibility_matrix[k, l], stem_compatibility_matrix[l, k] = 1.0,1.0
    
    crossing_matrix = generate_crossing_matrix(stems_s1, stems_s2, stem_indices)
    return(stems_s1, stems_s2, stem_indices, stem_compatibility_matrix, crossing_matrix)

def get_stems(bp_compatiable_matrix):
    """ identifies all stems in a fold from its base-pair compatibility matrix """
    stems_s1, stems_s2 = [],[]
    for i in range(0, bp_compatiable_matrix.shape[0]):
        js = get_off_diagonal(bp_compatiable_matrix, i)
        for j in js:
            if j > i:
                s1, s2 = trace_stem(bp_compatiable_matrix, i, j)
                stems_s1.append(s1)
                stems_s2.append(s2)
    return (stems_s1, stems_s2)


def trace_stem(bp_compatiable_matrix, i, j):
    """ Identifies individal stems by identifying uninterrupt base-pairs connect to i,j. 
        Such base-pairs radiate diagonally from i,j in bp_compatiable_matrix
    """
    stem_s1,stem_s2 = [],[]
    terminal = False
    while not terminal:
        if bp_compatiable_matrix[i,j] != 0:         
            stem_s1.append(i)
            stem_s2.append(j)
            i += 1
            j -= 1
        else:
            terminal = True
    return (stem_s1, stem_s2)

def get_off_diagonal(bp_compatiable_matrix, i):
    """ Get off-diagonal non-zero entries in a base-pair compatibility matrix 
        Useful for retriving the residues j that a given residue i can base-pair with
    """
    return(np.flatnonzero(bp_compatiable_matrix[i, :]))


def check_for_overlap(a, b):
    """ Returns true if two sets are not overlaping. Used here to check if two stems share a common residue. 
        If they do, returns False.    
    """
    # https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
    # return True is the is no overlap, else False
    return(set(a).isdisjoint(b))


def generate_crossing_matrix(stems_s1, stems_s2, stem_indices):
    """ Generates matrix that stores whether two stems are crossing 
        If crossing: 1
        If not crossing: 0
    """
    n = len(stem_indices)
    crossing_matrix = np.ones((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i < j and check_for_crossing([i, j], stems_s1, stems_s2):
                crossing_matrix[i, j], crossing_matrix[j, i] = 0.0, 0.0
    return(crossing_matrix)

def check_for_crossing(stems, stems_s1, stems_s2):
    """ Check if stems are crossing. Two stems are crossing if they have any two base-pair cross
        This function therefore loop over all combination of base-pairs can check if they cross.
        Two base-pairs cross if all following are satisfied:
           1) the residue index of 5'- residue in first base-pair is less than residue index of 5'- residue in second base-pair
           2) the residue index of 3'- residue in first base-pair is greater than residue index of 5'- residue in second base-pair
           3) the residue index of 3'- residue in first base-pair is less than residue index of 3'- residue in first base-pair
    """
    bp_i, bp_j = [], []
    for stem in stems:
        bp_i += stems_s1[stem]
        bp_j += stems_s2[stem]

    crossing = False
    loop_length = np.abs(np.array(bp_i)) - np.abs(np.array(bp_j))
    for i in range(0, len(bp_i)):
            for j in range(0, len(bp_i)):
                if i < j:
                    if bp_i[i] < bp_i[j] and bp_j[i] > bp_i[j] and bp_j[i] < bp_j[j]:
                        crossing = True
                        return(crossing)
                    if bp_i[i] > bp_i[j] and bp_j[i] > bp_j[j]:
                        crossing = True
                        return(crossing)
    return(crossing)

def check_compatibility(test_stem, reference_stems, compatibility_matrix):
    """ Checks if a given test stem is compatible with stems in another set of reference stems
        When folding RNA by adding stems, this function can used be to check if the added stem crosses or overlaps 
        the stems already present in the RNA
    """
    compatible = True
    for stem in reference_stems:
        if compatibility_matrix[stem, test_stem] == 0:
            compatible = False
            break
    return(compatible)

def get_stem_energies(stems_s1, stems_s2, HB_matrix, G_HB = -1.89, G_stack = -4.5):
    """ determine the folding energies for each stem:
        stems_s1: list of containing a list of residue indices associated with the 5' residues in each stem
        stems_s2: list of containing a list of residue indices associated with the 3' residues in each stem
        HB_matrix: the N x N matrix with counts of the hydrogen-bonds possible between pairs of residues
        G_HB: the energy contribution per hydrogen (default: -1.89 kcal/mol)
        G_stack: the energy contribution per stacking interaction (default: -4.5 kcal/mol)
    """
    stem_energies = []
    for index_a in range(0, len(stems_s1)):
        stem_1 = stems_s1[index_a]
        stem_2 = stems_s2[index_a]        
        # initialize energy
        ene = 0.0        
        # add base-pair energies
        for index_b in range(0, len(stem_1)):
            ene += HB_matrix[stem_1[index_b], stem_2[index_b]]*G_HB    
        # add stacking
        ene += (float(len(stem_1))-1)*G_stack
        stem_energies.append(ene)
    return(stem_energies)

def get_free_energy_from_stems(stems, stem_energies):
    """ determines the folding free energy from the set of assembled stems """
    free_energy = 0.0
    for stem in stems:
        free_energy += stem_energies[stem]
    return(free_energy)

def initialize_RNA(sequence = 'GGCACUUCGGUGCC', G_HB = -1.89, G_stack = -1.0):
    """ Initializes a dictionary that stores all the information needed fold an RNA,
        and which are derived from the sequence. 
        
        * A particular fold of an RNA is specified by a list of stems stored in: state['assembled_stems']
        
        * The list of stems indices and energies stored in: state['stems'] and state['stem_energies'], respectively
        
        * The matrix storing the compatibility of stems in stored in: state['stem_compatibility_matrix'] If two stems are not
          compatible, then they cannot appear together in a single fold
        
        * Similarly, the matrix storing crossings between stems is stored in: state['stem_crossing_matrix'] If two stems 
          cross each other, then they cannot appear together in a single fold.
        
    """
    state = {}        
    sequence = [i for i in sequence]
    residues = [ i for i in range(0, len(sequence))]

    bp_compatiable_matrix, hb_matrix = generate_basepair_compatibility_matrix(sequence)
    stems_s1, stems_s2, stem_indices, stem_compatibility_matrix, stem_crossing_matrix = setup_stems(bp_compatiable_matrix)
    stem_energies = get_stem_energies(stems_s1, stems_s2, hb_matrix, G_HB = G_HB, G_stack = G_stack)        

    stems = [i for i in range(0, len(stems_s1))]
    initial = random_select_i(stems)
    assembled_stems = []
    assembled_stems.append(initial)    
    
    state['sequence'] = sequence
    state['residues'] = residues
    state['bp_compatiable_matrix'] = bp_compatiable_matrix
    state['stems'] = stems
    state['stems_s1'] = stems_s1
    state['stems_s2'] = stems_s2
    state['stem_indices'] = stem_indices
    state['stem_compatibility_matrix'] = stem_compatibility_matrix
    state['stem_crossing_matrix'] = stem_crossing_matrix
    state['stem_energies'] = stem_energies
    state['assembled_stems'] = assembled_stems    
    return(state)

def perturb_stem(state):    
        """ Function modifies the stem in an RNA
        """        
        # chose a stem to modify
        modified = False
        while not modified:
            assembled_stems_tmp = copy.deepcopy(state['assembled_stems'])
            trial = random_select_i(state['stems'])            
            if trial in assembled_stems_tmp: # either remove from current structure            
                assembled_stems_tmp.remove(trial)
                modified = True
            else: # or add stem, if it compatible with stems in current structure            
                if check_compatibility(trial, assembled_stems_tmp, state['stem_compatibility_matrix']) and check_compatibility(trial, assembled_stems_tmp, state['stem_crossing_matrix']): 
                    assembled_stems_tmp.append(trial)
                    modified = True
        state['assembled_stems'] = copy.deepcopy(assembled_stems_tmp)
        return(state)
    
def states2averaged_base_matrix(states):
    """ Generates averaged base-paired matrix for list of states """
    bp_matrix = None
    nstates = len(states)
    for i in range(nstates):
        state = states[i]
        tmp = stem2basepair_matrix(state['sequence'], state['assembled_stems'], state['stems_s1'], state['stems_s2'])
        if bp_matrix is None:
            bp_matrix = tmp
        else:
            bp_matrix += tmp
    return(bp_matrix/nstates)
    