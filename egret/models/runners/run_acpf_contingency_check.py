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

def get_graph(md):
    G = nx.Graph()

    bus_attrs = md.attributes(element_type='bus')
    branches = dict(md.elements(element_type='branch'))

    G.add_nodes_from(bus_attrs['names'])
    G.add_edges_from([(branch['from_bus'], branch['to_bus']) for b, branch in branches.items() if branch['in_service']])

    return G

def solve_subgrid_acpf(md,subgrid):
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    loads = dict(md.elements(element_type='load'))
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)
    shunts = dict(md.elements(element_type='shunt'))
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    if len(subgrid) == 1:
        bus = buses[subgrid[0]]
        p_max = sum([gens[g]['p_max'] for g in gens_by_bus[subgrid[0]]])
        p_min = sum([gens[g]['p_min'] for g in gens_by_bus[subgrid[0]]])
        q_max = sum([gens[g]['q_max'] for g in gens_by_bus[subgrid[0]]])
        q_min = sum([gens[g]['q_min'] for g in gens_by_bus[subgrid[0]]])

        # solve single node balancing case
    else: pass
        # set a slack in the island
        # solve ACPF for the island
        # consolidate all the output data to a model_data object

if __name__ == '__main__':
    import os
    import networkx as nx
    import math
    from egret.parsers.matpower_parser import create_ModelData
    import random
    from egret.models.acpf import *
    from egret.models.acopf import *
    import sys
    import pyomo.opt as po

    random.seed(23) # repeatable

    run_islands = False

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
            md, m, results = solve_acopf(md, "ipopt", acopf_model_generator=create_psv_acopf_model, solver_tee=False,
                                         return_model=True, return_results=True, write_results=True,
                                         runid='sample.{}'.format(samples))

            if results.solver.termination_condition == po.TerminationCondition.optimal:
                break

        branches = dict(md.elements(element_type='branch'))
        for branch, branch_dict in branches.items():
            if branches[branch]['in_service'] == True:
                branches[branch]['in_service'] = False

                # check if graph is islanded
                graph = get_graph(md)
                if nx.is_connected(graph):
                    _, m, results = solve_acpf(md, "ipopt", solver_tee=False, return_model=True, return_results=True, write_results=True,
                                                    runid='sample.{}_branch.{}'.format(samples,branch))
                else:
                    if run_islands:
                        subgrids = [list(x) for x in nx.connected_components(graph)] # returns islands
                        for subgrid in subgrids:
                            solve_subgrid_acpf(md, subgrid)

                branches[branch]['in_service'] = True






