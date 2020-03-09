#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This varies the load and generator setpoints, then solves for ACPF.
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
    filename = 'pglib_opf_case118_ieee.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf-master/', filename)

    samples = 1
    max_samples = 100

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
                power_factor = load_dict['p_load']/math.sqrt(load_dict['p_load']**2 + load_dict['q_load']**2)
                load_dict['p_load'] = _variation_fraction*load_dict['p_load']
                load_dict['q_load'] = load_dict['p_load']*math.tan(math.acos(power_factor))

            for gen, gen_dict in gens.items():
                while True:
                    _variation_fraction = random.uniform(0.85,1.15)
                    p_tmp = gen_dict['pg']*_variation_fraction
                    if gen_dict['p_min'] <= p_tmp <= gen_dict['p_max']:
                        gen_dict['pg'] = p_tmp
                        break

            for bus, bus_dict in buses.items():
                _variation_fraction = random.uniform(bus_dict['v_min'],bus_dict['v_max'])
                bus_dict['vm'] = _variation_fraction

            md, m, results = solve_acpf(md, "ipopt", solver_tee=False, return_model=True, return_results=True, write_results=True,
                                        runid='sample.{}'.format(samples))

            if results.solver.termination_condition == po.TerminationCondition.optimal:
                samples += 1
                break






