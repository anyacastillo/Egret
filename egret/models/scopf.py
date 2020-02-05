#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module provides functions that create the modules for typical SCOPF formulations.

"""
import pyomo.environ as pe
import operator as op
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
import egret.model_library.decl as decl

from egret.models.acopf import create_psv_acopf_model, create_riv_acopf_model, create_rsv_acopf_model
from egret.model_library.defn import FlowType, CoordinateType
from egret.data.data_utils import map_items, zip_items
from math import pi, radians
from collections import OrderedDict


def create_scopf_model(model_data, contingency_dict, model_generator=create_psv_acopf_model, include_feasiblity_slack=True):

    #
    # Load model data
    #
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)
    ref_bus = md.data['system']['reference_bus']

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    loads = dict(md.elements(element_type='load'))

    bus_attrs = md.attributes(element_type='bus')
    buses_with_gens = tx_utils.buses_with_gens(gens)

    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    #
    # Create the pre-contingency (root) model
    #
    model = pe.ConcreteModel()
    model.pre_contingency_problem = model_generator(model_data, include_feasiblity_slack)

    #
    # Create indexed block of contingency subproblems for branch outages
    #
    model.contingency_branch_list = contingency_dict['branches']
    model.contingency_branch_subproblem = pe.Block(model.contingency_branch_list)

    #
    # Create indexed block of contingency subproblems for generator outages
    #
    model.contingency_gen_list = contingency_dict['generators']
    model.contingency_gen_subproblem = pe.Block(model.contingency_gen_list)

    #
    # Only defined for acopf formulations with vm (voltage magnitude) variable
    # Otherwise valid for all opf approximation formulations
    #
    if model_generator == create_rsv_acopf_model or model_generator == create_riv_acopf_model:
        model_generator = create_psv_acopf_model
    #
    # Generate operating model for each subproblem
    #
    for key, val in model.contingency_branch_dict:
        model.contingency_branch_subproblem[key] = model_generator(model_data, include_feasiblity_slack)

    #
    # Linking voltage constraints for root model to each subproblem
    #
    if hasattr(model.pre_contingency_problem,'vm'): # need to check whether this is a dcopf approximation or not
        con_set_voltage = decl.declare_set('_con_eq_gen_voltage', model, model.contingency_branch_list,
                                           bus_attrs['names'])
        model.eq_gen_voltage = pe.Constraint(con_set_voltage)

        for key, bus_name in con_set_voltage:
            if bus_p_loads[bus_name] != 0.0 or bus_q_loads[bus_name] != 0.0:
                model.eq_gen_voltage[key, bus_name] = \
                    model.pre_contingency_problem.vm[bus_name] == model.contingency_branch_subproblem[key].vm[bus_name]

    #
    # Linking real power generation constraints for root model to each subproblem
    #
    con_set_power = decl.declare_set('_con_eq_gen_power', model, model.contingency_branch_list, bus_attrs['names'])
    model.eq_gen_power = pe.Constraint(con_set_power)

    for key, bus_name in con_set_power:
        if bus_name in buses_with_gens and bus_name != ref_bus:
            model.eq_gen_power[key, bus_name] = \
                model.pre_contingency_problem.pg[bus_name] == model.contingency_branch_subproblem[key].pg[bus_name]

    return model

def solve_scopf(model_data,
                contingency_dict,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                scopf_model_generator = create_psv_acopf_model,
                return_model = False,
                return_results = False,
                **kwargs):
    '''
    Create and solve a new scopf model

    Parameters
    ----------
    model_data : egret.data.ModelData
        An egret ModelData object with the appropriate data loaded.
    contingency_dict : dict
        A dictionary of lists for contingencies, e.g., {'branches': ['1','8','23','25','37','44'], 'generators': ['2']}
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instantiated pyomo solver
    timelimit : float (optional)
        Time limit for dcopf run. Default of None results in no time
        limit being set.
    solver_tee : bool (optional)
        Display solver log. Default is True.
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels. Useful for debugging; default is False.
    options : dict (optional)
        Other options to pass into the solver. Default is dict().
    scopf_model_generator : function (optional)
        Function for generating the acopf model. Default is
        egret.models.acopf.create_psv_acopf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    from pyomo.environ import value
    from egret.common.solver_interface import _solve_model
    from egret.model_library.transmission.tx_utils import \
        scale_ModelData_to_pu, unscale_ModelData_to_pu

    if kwargs is None:
        kwargs = {'include_feasibility_slack': True}

    m, md = scopf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    m, results = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,solver_options=options)

    md.data['system']['total_cost'] = value(m.pre_contingency_problem.obj)

    unscale_ModelData_to_pu(md, inplace=True)

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case39_epri.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    model_data = create_ModelData(matpower_file)
    contingency_dict = {'branches': ['1','8','23','25','37','44']}
    md,m,results = solve_scopf(model_data, contingency_dict, "ipopt",scopf_model_generator=create_psv_acopf_model,return_model=True, return_results=True)


