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


def create_fictional_slack_gen(md, _md):
    # only consider buses in the subgrid, represented by _md modeldata object
    buses = dict(_md.elements(element_type='bus'))

    md.data['elements']['generator']['fictitious'] = dict()
    md.data['elements']['generator']['fictitious']['bus'] = list(buses.keys())[0]
    md.data['elements']['generator']['fictitious']['pg'] = 0.
    md.data['elements']['generator']['fictitious']['qg'] = 0.
    md.data['elements']['generator']['fictitious']['vg'] = 1.0
    md.data['elements']['generator']['fictitious']['mbase'] = 100.0
    md.data['elements']['generator']['fictitious']['in_service'] = True
    md.data['elements']['generator']['fictitious']['p_max'] = 0.
    md.data['elements']['generator']['fictitious']['p_min'] = 0.
    md.data['elements']['generator']['fictitious']['q_max'] = 0.
    md.data['elements']['generator']['fictitious']['q_min'] = 0.
    md.data['elements']['generator']['fictitious']['generator_type'] = 'thermal'
    md.data['elements']['generator']['fictitious']['p_cost'] = dict()
    md.data['elements']['generator']['fictitious']['p_cost']['data_type'] = 'cost_curve'
    md.data['elements']['generator']['fictitious']['p_cost']['cost_curve_type'] = 'polynomial'
    md.data['elements']['generator']['fictitious']['p_cost']['values'] = {0: 0., 1: 0., 2: 0.}
    md.data['elements']['generator']['fictitious']['p_cost']['startup_cost'] = 0.0
    md.data['elements']['generator']['fictitious']['p_cost']['shuhtdown_cost'] = 0.0

    _md.data['elements']['generator']['fictitious'] = md.data['elements']['generator']['fictitious']


def get_modeldata_subgrid(md,subgrid):
    # clonse the full grid md modeldata object
    _md = md.clone()

    buses = dict(md.elements(element_type='bus'))
    generators = dict(md.elements(element_type='generator'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    # remove from the clone _md any buses that don't appear in the subgrid
    for bus, bus_dict in buses.items():
        if bus not in subgrid:
            del _md.data['elements']['bus'][bus]

    # remove from the clone _md any gens that don't appear in the subgrid
    for gen, gen_dict in generators.items():
        if gen_dict['bus'] not in subgrid:
            del _md.data['elements']['generator'][gen]

    # remove from the clone _md any loads that don't appear in the subgrid
    for load, load_dict in loads.items():
        if load_dict['bus'] not in subgrid:
            del _md.data['elements']['load'][load]

    # remove from the clone _md any shunts that don't appear in the subgrid
    for shunt, shunt_dict in shunts.items():
        if shunt_dict['bus'] not in subgrid:
            del _md.data['elements']['shunt'][shunt]

    # remove from the clone _md any branches that don't appear in the subgrid
    for branch, branch_dict in branches.items():
        from_bus = branch_dict['from_bus']
        to_bus = branch_dict['to_bus']
        if (from_bus not in subgrid) or (to_bus not in subgrid):
            del _md.data['elements']['branch'][branch]

    return _md


def solve_subgrid_acpf(md,_md):
    _gens = dict(md.elements(element_type='generator'))
    _buses = dict(md.elements(element_type='bus'))
    _branches = dict(md.elements(element_type='branch'))

    # first set a reference bus for the subgrid
    _bus_attrs = _md.attributes(element_type='bus')
    ref_bus = _md.data['system']['reference_bus']
    if not ref_bus in _bus_attrs['names']:
        pv_buses = [bus for bus, bus_dict in _buses.items() if bus_dict['matpower_bustype'] == 'PV']
        if not pv_buses is None:
            # if there is a PV bus, set this as the new slack bus
            _md.data['system']['reference_bus'] = pv_buses[0]
        else:
            # if there is no PV bus (all load buses), create a fictitious generator at an arbitrary slack bus
            create_fictional_slack_gen(md, _md)

    if len(subgrid) == 1:
        # solve single node balancing case
        from egret.models.copperplate_dispatch import solve_copperplate_dispatch, create_copperplate_ac_approx_model
        _md, m, results = solve_copperplate_dispatch(_md, solver='ipopt', solver_tee = False, copperplate_dispatch_model_generator = create_copperplate_ac_approx_model, return_model = True, return_results = True)
    else:
        # solve the acpf for the island
        _md, m, results = solve_acpf(_md, "ipopt", solver_tee=False, return_model=True, return_results=True,
                                   write_results=False)

    # save results data for solving the subgrid to the md modeldata object for the full grid
    md.data['system']['total_cost'] += _md.data['system']['total_cost']

    for g,g_dict in _gens.items():
        md.data['elements']['generator'][g]['pg'] = g_dict['pg']
        md.data['elements']['generator'][g]['qg'] = g_dict['qg']

    for b,b_dict in _buses.items():
        md.data['elements']['bus'][b]['lmp'] = b_dict['lmp']
        md.data['elements']['bus'][b]['qlmp'] = b_dict['qlmp']
        md.data['elements']['bus'][b]['pl'] = b_dict['pl']
        md.data['elements']['bus'][b]['vm'] = b_dict['vm']
        md.data['elements']['bus'][b]['va'] = b_dict['va']

    for k, k_dict in _branches.items():
        if hasattr(k_dict,'pf'):
            md.data['elements'][k]['pf'] = k_dict['pf']
            md.data['elements'][k]['pt'] = k_dict['pt']
            md.data['elements'][k]['qf'] = k_dict['qf']
            md.data['elements'][k]['qt'] = k_dict['qt']

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

    run_islands = True

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case14_ieee.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf-master/', filename)

    samples = 0
    max_samples = 10

    # if len(sys.argv[1:]) == 1:
    #     max_samples = sys.argv[1] # argument 1: # of samples
    # if len(sys.argv[1:]) == 2:
    #     max_samples = sys.argv[1] # argument 1: # of samples
    #     filename = sys.argv[2] # argument 2: case filename

    while samples < max_samples:

        # increment samples computed and saved to json
        samples += 1

        while True:

            # N-1 contingency (line outages only)
            md = create_ModelData(matpower_file)
            loads = dict(md.elements(element_type='load'))

            # randomize real (p) and reactive (q) loads, maintaining a power factor p/sqrt(p^2 + q^2)
            for load, load_dict in loads.items():
                _variation_fraction = random.uniform(0.85,1.15)
                power_factor = load_dict['p_load']/math.sqrt(load_dict['p_load']**2 + load_dict['q_load']**2)
                load_dict['p_load'] = _variation_fraction*load_dict['p_load']
                load_dict['q_load'] = load_dict['p_load']*math.tan(math.acos(power_factor))

            # solve ACOPF to determine the optimal control variables for this operating point
            kwargs = {'include_feasibility_slack': False}
            md, m, results = solve_acopf(md, "ipopt", acopf_model_generator=create_psv_acopf_model, solver_tee=False,
                                         return_model=True, return_results=True, write_results=True,
                                         runid='sample.{}'.format(samples))

            # if the ACOPF solve was infeasible, try again
            if results.solver.termination_condition == po.TerminationCondition.optimal:
                break

        # if the ACOPF solve was optimal, then do a full N-1 contingency analysis (on all the branches)
        # for this operating point
        branches = dict(md.elements(element_type='branch'))
        for branch, branch_dict in branches.items():
            # if branch is in-service, then perform contingency analysis on it
            if branches[branch]['in_service'] == True:
                # set contingency branch out-of-service
                branches[branch]['in_service'] = False

                # check if graph is islanded
                graph = get_graph(md)
                if nx.is_connected(graph):
                    # set the 'islanded' paramter to None since there are no islands
                    md.data['system']['islanded'] = None
                    # solve acpf for the full grid
                    _, m, results = solve_acpf(md, "ipopt", solver_tee=False, return_model=True, return_results=True, write_results=True,
                                                    runid='sample.{}_branch.{}'.format(samples,branch))
                else:
                    if run_islands:
                        # determine nodes in each subgrid
                        subgrids = [list(x) for x in nx.connected_components(graph)] # returns islands
                        # set system cost to 0.
                        md.data['system']['total_cost'] = 0.
                        # save subgrid partitions (list of lists) to modeldata object
                        md.data['system']['islanded'] = subgrids
                        # solve acpf/balance for each island
                        for subgrid in subgrids:
                            # create a _md modeldata object to represent the subgrid only
                            _md = get_modeldata_subgrid(md, subgrid)
                            # solve acpf/balance for _md, and store results back into md modeldata object for the full grid
                            solve_subgrid_acpf(md, _md)
                        # write the md modeldata object to json after all subgrids are solved
                        system = md.data['system']['model_name']
                        runid = 'sample.{}_branch.{}'.format(samples, branch)
                        filename = "%s__acpf_runid_%s.json" % (system, str(runid))
                        md.write(filename, file_type='json')

                # set branch back to in-service
                branches[branch]['in_service'] = True






