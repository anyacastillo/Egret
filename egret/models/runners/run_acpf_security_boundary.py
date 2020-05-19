#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This implements the efficient dataset creation approach in:
Efficient Creation of Datasets for Data-Driven Power System Applications
A. Venzke, D. K. Molzahn, S. Chatzivasileiadis
arXiv:1910.01794v1 [eess.SY] 4 Oct 2019
"""

if __name__ == '__main__':
    import math
    from egret.parsers.matpower_parser import create_ModelData
    import random
    from egret.models.acpf import *
    from egret.models.acopf import *
    import sys
    import pyomo.opt as po
    from egret.models.ac_relaxations import create_acopf_security_boundary
    import os

    random.seed(23) # repeatable

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case118_ieee.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf-master/', filename)

    samples = 1
    max_samples = 5

    if len(sys.argv[1:]) == 1:
        max_samples = int(sys.argv[1]) # argument 1: # of samples
    if len(sys.argv[1:]) == 2:
        max_samples = int(sys.argv[1]) # argument 1: # of samples
        filename = sys.argv[2] # argument 2: case filename

    while samples <= max_samples:

        while True:
            md = create_ModelData(matpower_file)
            loads = dict(md.elements(element_type='load'))
            gens = dict(md.elements(element_type='generator'))
            buses = dict(md.elements(element_type='bus'))

            for load, load_dict in loads.items():
                _variation_fraction = random.uniform(0.85,1.15)
                power_factor = random.uniform(0.5,1.)
                load_dict['p_load'] = _variation_fraction*load_dict['p_load']
                load_dict['q_load'] = load_dict['p_load']*math.tan(math.acos(power_factor))

            for gen, gen_dict in gens.items():
                _variation_fraction = random.uniform(0.85,1.15)
                power_factor = random.uniform(0.5,1.)
                gen_dict['pg'] = _variation_fraction*gen_dict['pg']
                gen_dict['qg'] = gen_dict['pg']*math.tan(math.acos(power_factor))

            for bus, bus_dict in buses.items():
                _variation_fraction = random.uniform(bus_dict['v_min']*0.9,bus_dict['v_max']*1.1)
                bus_dict['vm'] = _variation_fraction

            model, md = create_acopf_security_boundary(md)

            from egret.common.solver_interface import _solve_model
            m, results = _solve_model(model, "ipopt")#, timelimit=timelimit, solver_tee=solver_tee,
                                      #symbolic_solver_labels=symbolic_solver_labels, solver_options=options)

            if results.solver.termination_condition == po.TerminationCondition.optimal:
                samples += 1
                break
