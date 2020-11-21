import network_line_graph as nlg


from annealing import *
from PyRNA import *

# dictionary to store results
results = {}


results['TAR'] = {}
results['TAR']['bp'], results['TAR']['ct'] = CT2basepair_matrix("data/7JU1.ct")


results['TAR']['states'] = simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=3,
                    cooling_rate = 0.95,
                    cooling_steps = 1000,
                    equilibration_steps = 3,
                    distribution_parameter = 5.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")



visualize_structure(states2averaged_base_matrix(results['TAR']['states']), label = "Average")
visualize_structure(results['TAR']['bp'], label = "Acutal")


import joblib
filename = 'data/states_HIV_TAR.sav'
joblib.dump(results, filename)


state = results['TAR']['states'][0]
visualize_structure(state2basepair_matrix(state))
state2CT(state)
