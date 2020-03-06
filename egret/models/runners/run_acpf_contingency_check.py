#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This runs the ACOPF for the current operating point,
and then for a given contingency, checks to see if the operating point
is AC power flow feasible
"""

if __name__ == '__main__':
    import os
    import math
    from egret.parsers.matpower_parser import create_ModelData
    import random
    from egret.models.acpf import *
    from egret.models.acopf import *
    import sys
    import pyomo.opt as po

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

        while True:
            # N-1 contingency (line outages only
            md = create_ModelData(matpower_file)
            loads = dict(md.elements(element_type='load'))

            for load, load_dict in loads.items():
                _variation_fraction = random.uniform(0.85,1.15)
                power_factor = load_dict['p_load']/math.sqrt(load_dict['p_load']**2 + load_dict['q_load']**2)
                load_dict['p_load'] = _variation_fraction*load_dict['p_load']
                load_dict['q_load'] = load_dict['p_load']*math.tan(math.acos(power_factor))

            kwargs = {'include_feasibility_slack': False}
            md, m, results = solve_acopf(md, "ipopt", acopf_model_generator=create_psv_acopf_model, return_model=True,
                                         return_results=True, write_results=True,
                                         runid='sample.{}'.format(samples))

            if results.solver.termination_condition == po.TerminationCondition.optimal:
                samples += 1
                break

        branches = dict(md.elements(element_type='branch'))
        for branch, branch_dict in branches.items():
            if branches[branch]['in_service'] == True:
                branches[branch]['in_service'] = False
                md, m, results = solve_acpf(md, "ipopt", return_model=True, return_results=True, write_results=True,
                                                runid='sample.{}_branch.{}'.format(samples,branch))
                branches[branch]['in_service'] = True






