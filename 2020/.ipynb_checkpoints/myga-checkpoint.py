import numpy as np
import random
from itertools import groupby
from operator import itemgetter
import copy
from PyRNA import *
from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga    
import network_line_graph as nlg

def run_genetic_algorithm(sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC', N_stems = 4, iterations = 10, population_size = 10):
    """ Function using a genetic algorithim to fold an RNA """
    # Intialize state
    initial_state = initialize_RNA(sequence, G_HB = -1.89, G_stack = -20.0)
    stem_energies = initial_state['stem_energies']
    stem_compatibility_matrix = initial_state['stem_compatibility_matrix']
    stem_crossing_matrix = initial_state['stem_crossing_matrix']
    varbound=np.array([[-1, len(initial_state['stem_indices'])-1]]*N_stems)
    algorithm_param = {'max_num_iteration': 5000,\
                       'population_size':population_size,\
                       'mutation_probability':0.1,\
                       'elit_ratio': 0.01,\
                       'crossover_probability': 0.5,\
                       'parents_portion': 0.3,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None}
    
    def fitness(X):      
        """ Fitness Function """
        ene = 0
        assembled_stems = []
        filtered_X = [] 
        [filtered_X.append(int(x)) for x in X if x not in filtered_X] 
        for x in filtered_X:
            if x == -1:
                continue
            else:
                assembled_stems.append(x)

        valid_structure = False
        for trial in assembled_stems:
            assembled_stems_tmp = copy.deepcopy(assembled_stems)
            assembled_stems_tmp.remove(trial)
            valid_structure = check_compatibility(trial, assembled_stems_tmp, stem_compatibility_matrix) and check_compatibility(trial, assembled_stems_tmp, stem_crossing_matrix)
            if not valid_structure:
                break
        ene = 0.0
        if valid_structure:
                for x in assembled_stems:
                    ene = ene + stem_energies[x]
        else:
            ene = 99.9

        return(ene)
    
    # use GA to generate low-energy states
    models, states = [],[]
    for i in range(iterations):
        model=ga(function=fitness, 
                 dimension=N_stems,
                 variable_type='int', 
                 variable_boundaries=varbound, 
                 algorithm_parameters=algorithm_param)
        model.run()
        initial_state['assembled_stems'] = ga2stems(model)
        visualize_structure(state2basepair_matrix(initial_state), "GA iteration: %s"%(i))
        models.append(copy.deepcopy(model))
        states.append(copy.deepcopy(initial_state))
    return(models, states)