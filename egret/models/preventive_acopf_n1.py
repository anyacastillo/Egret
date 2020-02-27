#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
Preventive ACOPF N-1 Contingency Analysis

Efficient Creation of Datasets for Data-Driven Power System Applications
Andreas Venzke, Daniel K. Molzahn and Spyros Chatzivasileiadis
arXiv:1910.01794v1
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


def create_acopf_n1_model(model_data, contingency_dict, model_generator=create_psv_acopf_model, include_feasiblity_slack=True):

    #
    # Load model data
    #
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)
    ref_bus = md.data['system']['reference_bus']

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    loads = dict(md.elements(element_type='load'))
    branches = dict(model_data.elements(element_type='branch'))

    bus_attrs = md.attributes(element_type='bus')
    gen_attrs = md.attributes(element_type='generator')
    buses_with_gens = tx_utils.buses_with_gens(gens)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    #
    # Create model and contingency lists
    #
    model = pe.ConcreteModel()
    model.contingency_branch_list = list()
    model.contingency_gen_list = list()
    if 'branches' in contingency_dict:
        model.contingency_branch_list = ['branch_'+ i for i in contingency_dict['branches']]
    if 'generators' in contingency_dict:
        model.contingency_gen_list = ['gen_' + i for i in contingency_dict['generators']]
    model.contingency_list = model.contingency_branch_list + model.contingency_gen_list
    if len(model.contingency_list) == 0:
        raise Exception("No contingencies specified.")
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
        model_generator(model_data, model.pre_contingency_problem[con], include_feasiblity_slack)
        _type, _idx = con.split('_')
        if _type == "branch":
            if branches[_idx]['in_service'] == True:
                branches[_idx]['in_service'] = False
                model_generator(model_data, model.post_contingency_problem[con], include_feasiblity_slack)
                delattr(model.post_contingency_problem[con],'obj')
                branches[_idx]['in_service'] = True
        if _type == "gen":
            if gens[_idx]['in_service'] == True:
                gens[_idx]['in_service'] = False
                model_generator(model_data, model.post_contingency_problem[con], include_feasiblity_slack)
                delattr(model.post_contingency_problem[con],'obj')
                gens[_idx]['in_service'] = True

    #
    # Linking voltage constraints for root model to each subproblem
    #
    if hasattr(model.pre_contingency_problem[_idx0],'vm'): # need to check whether this is a dcopf approximation or not
        bus_set = tx_utils.buses_with_gens(gens)
        index_set = list()
        for con_name in model.contingency_list:
            for bus_name in bus_set:
                index_set.append((con_name, bus_name))
        con_set_voltage = decl.declare_set('_con_eq_gen_voltage', model, index_set)
        model.eq_gen_voltage = pe.Constraint(con_set_voltage)
        model.eq_gen_voltage_nonanticipative = pe.Constraint(con_set_voltage)

        for con, bus_name in con_set_voltage:
            model.eq_gen_voltage[con, bus_name] = \
                    model.pre_contingency_problem[con].vm[bus_name] == model.post_contingency_problem[con].vm[bus_name]
            if con != _idx0:
                model.eq_gen_voltage_nonanticipative[con, bus_name] = \
                        model.pre_contingency_problem[_idx0].vm[bus_name] == model.pre_contingency_problem[con].vm[bus_name]

    #
    # Linking real power generation constraints for root model to each subproblem
    #
    index_set = list()
    for con_name in model.contingency_list:
        for gen_name in gen_attrs['names']:
            index_set.append((con_name, gen_name))
    con_set_power = decl.declare_set('_con_eq_gen_power', model, index_set)
    model.eq_gen_power = pe.Constraint(con_set_power)
    model.eq_gen_power_nonanticipative = pe.Constraint(con_set_power)

    for con, gen_name in con_set_power:
        model.eq_gen_power[con, gen_name] = \
            model.pre_contingency_problem[con].pg[gen_name] == model.post_contingency_problem[con].pg[gen_name]
        if con != _idx0:
            model.eq_gen_voltage_nonanticipative[con, gen_name] = \
                        model.pre_contingency_problem[_idx0].pg[gen_name] == model.pre_contingency_problem[con].pg[gen_name]

    return model, md

def solve_acopf_n1(model_data,
                contingency_dict,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                acopf_n1_model_generator = create_psv_acopf_model,
                return_model = False,
                return_results = False,
                write_results = False,
                runid = '',
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
    acopf_n1_model_generator : function (optional)
        Function for generating the acopf model. Default is
        egret.models.acopf.create_psv_acopf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    write_results : bool (optional)
        If True, writes results of model_data object to json file
    runid : str
        If None, uses datetimestamp to name file when write_results is true; otherwise uses string specified here.
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

    model, md = create_acopf_n1_model(model_data, contingency_dict, model_generator=acopf_n1_model_generator, **kwargs)

    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    model, results = _solve_model(model,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,solver_options=options)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    _idx0 = model.contingency_list[0]
    m = model.pre_contingency_problem[_idx0]
    md.data['system']['total_cost'] = value(m.obj)

    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    for b,b_dict in buses.items():
        #b_dict['lmp'] = value(m.dual[m.eq_p_balance[b]])
        #b_dict['qlmp'] = value(m.dual[m.eq_q_balance[b]])
        b_dict['pl'] = value(m.pl[b])
        if hasattr(m,'p_slack_pos'):
            b_dict['p_slack_pos'] = value(m.p_slack_pos[b])
        if hasattr(m, 'p_slack_neg'):
            b_dict['p_slack_neg'] = value(m.p_slack_neg[b])
        if hasattr(m, 'q_slack_pos'):
            b_dict['q_slack_pos'] = value(m.q_slack_pos[b])
        if hasattr(m, 'q_slack_neg'):
            b_dict['q_slack_neg'] = value(m.q_slack_neg[b])
        if hasattr(m, 'vj'):
            b_dict['vm'] = tx_calc.calculate_vm_from_vj_vr(value(m.vj[b]), value(m.vr[b]))
            b_dict['va'] = tx_calc.calculate_va_from_vj_vr(value(m.vj[b]), value(m.vr[b]))
        else:
            b_dict['vm'] = value(m.vm[b])
            b_dict['va'] = value(m.va[b])

    for k, k_dict in branches.items():
        if hasattr(m,'pf'):
            k_dict['pf'] = value(m.pf[k])
            k_dict['pt'] = value(m.pt[k])
            k_dict['qf'] = value(m.qf[k])
            k_dict['qt'] = value(m.qt[k])
        if hasattr(m,'irf'):
            b = k_dict['from_bus']
            k_dict['pf'] = value(tx_calc.calculate_p(value(m.ifr[k]), value(m.ifj[k]), value(m.vr[b]), value(m.vj[b])))
            k_dict['qf'] = value(tx_calc.calculate_q(value(m.ifr[k]), value(m.ifj[k]), value(m.vr[b]), value(m.vj[b])))
            b = k_dict['to_bus']
            k_dict['pt'] = value(tx_calc.calculate_p(value(m.itr[k]), value(m.itj[k]), value(m.vr[b]), value(m.vj[b])))
            k_dict['qt'] = value(tx_calc.calculate_q(value(m.itr[k]), value(m.itj[k]), value(m.vr[b]), value(m.vj[b])))

    unscale_ModelData_to_pu(md, inplace=True)

    if write_results:
        system = model_data.data['system']['model_name']
        if runid is None:
            from datetime import datetime
            runid = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        filename = "%s__runid_%d.json" % (system, runid)
        md.write(filename,file_type='json')

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
    filename = 'pglib_opf_case14_ieee.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    model_data = create_ModelData(matpower_file)

    # contingency_dict format example: {'branches': ['1','8','23','25','37','44']}
    contingency_dict = {'branches': ['1','2','3']}
    md, m, results = solve_acopf_n1(model_data, contingency_dict, "ipopt",acopf_n1_model_generator=create_psv_acopf_model,return_model=True, return_results=True, write_results=True)


