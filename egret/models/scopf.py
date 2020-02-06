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
    gen_attrs = md.attributes(element_type='bus')
    buses_with_gens = tx_utils.buses_with_gens(gens)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    #
    # Create model and contingency lists
    #
    model = pe.ConcreteModel()
    model.contingency_branch_list = [('branch',i) for i in contingency_dict['branches']]
    model.contingency_gen_list = [('gen',i) for i in contingency_dict['generators']]
    model.contingency_list = model.contingency_branch_list + model.contingency_gen_list
    _idx0 = model.contingency_list[0]

    #
    # Create the pre-contingency (root) model
    #
    model.pre_contingency_problem = pe.Block(model.contingency_list)

    #
    # Create indexed block of contingency subproblems for branch and generator outages
    #
    model.post_contingency_problem = pe.Block(model.contingency_list)

    #
    # Only defined for acopf formulations with vm (voltage magnitude) variable
    # Otherwise valid for all opf approximation formulations
    #
    if model_generator == create_rsv_acopf_model or model_generator == create_riv_acopf_model:
        model_generator = create_psv_acopf_model
    #
    # Generate operating model for each subproblem
    #
    for con in model.contingency_list:
        model.pre_contingency_problem[con] = model_generator(model_data, include_feasiblity_slack)
        (_type, _idx) = con
        if _type == "branch": pass
        if _type == "gen": pass
        model.post_contingency_problem[con] = model_generator(model_data, include_feasiblity_slack)


    #
    # Linking voltage constraints for root model to each subproblem
    #
    if hasattr(model.pre_contingency_problem[_idx0],'vm'): # need to check whether this is a dcopf approximation or not
        con_set_voltage = decl.declare_set('_con_eq_gen_voltage', model, model.contingency_list,
                                           bus_attrs['names'])
        model.eq_gen_voltage = pe.Constraint(con_set_voltage)
        model.eq_gen_voltage_nonanticipative = pe.Constraint(con_set_voltage)

        for con, bus_name in con_set_voltage:
            if bus_p_loads[bus_name] != 0.0 or bus_q_loads[bus_name] != 0.0:
                model.eq_gen_voltage[con, bus_name] = \
                    model.pre_contingency_problem[con].vm[bus_name] == model.post_contingency_problem[con].vm[bus_name]
                if con != _idx0:
                    model.eq_gen_voltage_nonanticipative[con, bus_name] = \
                        model.pre_contingency_problem[_idx0].vm[bus_name] == model.pre_contingency_problem[con].vm[bus_name]

    #
    # Linking real power generation constraints for root model to each subproblem
    #
    con_set_power = decl.declare_set('_con_eq_gen_power', model, model.contingency_list, gen_attrs['names'])
    model.eq_gen_power = pe.Constraint(con_set_power)
    model.eq_gen_power_nonanticipative = pe.Constraint(con_set_power)

    for con, bus_name in con_set_power:
        if bus_name in buses_with_gens and bus_name != ref_bus:
            for gen_name in gens_by_bus[bus_name]:
                model.eq_gen_power[con, bus_name] = \
                    model.pre_contingency_problem[con].pg[gen_name] == model.post_contingency_problem[con].pg[gen_name]
                if con != _idx0:
                    model.eq_gen_voltage_nonanticipative[con, bus_name] = \
                        model.pre_contingency_problem[_idx0].pg[gen_name] == model.pre_contingency_problem[con].pg[gen_name]

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


