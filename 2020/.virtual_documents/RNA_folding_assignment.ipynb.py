#get_ipython().getoutput("pip install git+https://github.com/sbottaro/BME.git")
get_ipython().getoutput("git clone https://github.com/paulbrodersen/network_line_graph.git")
get_ipython().run_line_magic("cd", " network_line_graph/")
import network_line_graph as nlg


from annealing_assignment import *
from PyRNA import *


simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCACUUCGGUGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=1000,
                    cooling_rate = 0.95,
                    cooling_steps = 1000,
                    equilibration_steps = 10,
                    distribution_parameter = 10.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")


simulated_annealing(initial_state = initialize_RNA(sequence = 'GGCUUUUGCUCCCCGUGCUUCGGCACGGAAAAGCC', G_HB = -1.89, G_stack = -1.0),
                    iterations=1000,
                    cooling_rate = 0.95,
                    cooling_steps = 1000,
                    equilibration_steps = 10,
                    distribution_parameter = 10.0,
                    temperature = 300*0.001985875,
                    debug=False,                     
                    system = "RNA")


# Type your answer here
