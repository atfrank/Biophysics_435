import network_line_graph as nlg


from annealing import *
from PyRNA import *

# dictionary to store results
results = {}


results['TAR'] = {}
results['TAR']['bp'], results['TAR']['ct'] = bp, ct = CT2basepair_matrix("data/7JU1.ct")
results['TAR']['states'] = simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=100,
                    cooling_rate = 0.95,
                    cooling_steps = 1000,
                    equilibration_steps = 20,
                    distribution_parameter = 10.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")


results['tetraloop'] = {}
results['tetraloop']['bp'], results['tetraloop']['ct'] = bp, ct = CT2basepair_matrix("data/2KOC.ct")
results['tetraloop']['states'] = simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCACUUCGGUGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=10,
                    cooling_rate = 0.95,
                    cooling_steps = 10,
                    equilibration_steps = 20,
                    distribution_parameter = 10.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")


results['telomerase'] = {}
results['telomerase']['bp'], results['telomerase']['ct'] = bp, ct = CT2basepair_matrix("data/2L3E.ct")
results['telomerase']['states'] = simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCUUUUGCUCCCCGUGCUUCGGCACGGAAAAGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=100,
                    cooling_rate = 0.95,
                    cooling_steps = 2000,
                    equilibration_steps = 20,
                    distribution_parameter = 10.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")


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
 


results['tetraloop']['bp']


ave_bp = states2averaged_base_matrix(results['tetraloop']['states'])
act_bp = results['tetraloop']['bp']
visualize_structure(ave_bp, label = "Average")
visualize_structure(act_bp, label = "Acutal")


visualize_structure(states2averaged_base_matrix(results['telomerase']['states']), label = "Average")
visualize_structure(results['telomerase']['bp'], label = "Acutal")
