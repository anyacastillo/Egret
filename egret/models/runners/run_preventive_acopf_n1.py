#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
Running the Preventive ACOPF N-1 Contingency Analysis

Efficient Creation of Datasets for Data-Driven Power System Applications
Andreas Venzke, Daniel K. Molzahn and Spyros Chatzivasileiadis
arXiv:1910.01794v1
"""

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData
    import random
    from egret.models.preventive_acopf_n1 import *
    import sys
    import math

    random.seed(23) # repeatable

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case14_ieee.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf-master/', filename)

    samples = 0
    max_samples = 10

    if len(sys.argv[1:]) == 1:
        max_samples = sys.argv[1] # argument 1: # of samples
    if len(sys.argv[1:]) == 2:
        max_samples = sys.argv[1] # argument 1: # of samples
        filename = sys.argv[2] # argument 2: case filename

    while samples < max_samples:
        samples += 1

        # N-1 contingency (line outages only
        model_data = create_ModelData(matpower_file)
        branches = dict(model_data.elements(element_type='branch'))
        loads = dict(model_data.elements(element_type='load'))

        for branch, branch_dict in branches.items():
            contingency_dict = {'branches': [branch]}

        for load, load_dict in loads.items():
            _variation_fraction = random.uniform(0.85,1.15)
            power_factor = load_dict['p_load']/math.sqrt(load_dict['p_load']**2 + load_dict['q_load']**2)
            load_dict['p_load'] = _variation_fraction*load_dict['p_load']
            load_dict['q_load'] = load_dict['p_load']*math.tan(math.acos(power_factor))

            # contingency_dict format example: {'branches': ['1','8','23','25','37','44']}
            md, m, results = solve_acopf_n1(model_data, contingency_dict, "ipopt",acopf_n1_model_generator=create_psv_acopf_model, solver_tee=False, return_model=True, return_results=True, write_results=True, runid=samples)



