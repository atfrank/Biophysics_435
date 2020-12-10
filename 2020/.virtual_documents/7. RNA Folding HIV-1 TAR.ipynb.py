from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga
import network_line_graph as nlg
from myga import run_genetic_algorithm
from PyRNA import initialize_RNA, CT2basepair_matrix, check_compatibility, ga2stems, state2basepair_matrix, visualize_structure
import numpy as np

# dictionary to store results
results = {}


results['TAR'] = {}
results['TAR']['bp'], results['TAR']['ct'] = CT2basepair_matrix("data/7JU1.ct")
results['TAR']['models'], results['TAR']['states'] = run_genetic_algorithm(sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC', 
                                                                           N_stems = 4, 
                                                                           iterations = 200, population_size=30)


from PyRNA import states2averaged_base_matrix
visualize_structure(states2averaged_base_matrix(results['TAR']['states']), label = "Average")
visualize_structure(results['TAR']['bp'], label = "Acutal")


import joblib
filename = 'data/states_HIV_TAR.sav'
joblib.dump(results['TAR']['states'], filename)


from PyRNA import state2CT
state = results['TAR']['states'][195]
visualize_structure(state2basepair_matrix(state))
state2CT(state)
