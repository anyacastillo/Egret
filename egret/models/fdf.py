#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module provides functions that create the modules for typical ACOPF formulations.

#TODO: document this with examples
"""
import sys
import pyomo.environ as pe
from math import inf, sqrt
import pandas as pd
from egret.common.log import logger
import logging
import egret.models.tests.test_approximations as test
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
from egret.models.acopf import solve_acopf
from egret.model_library.defn import ApproximationType, SensitivityCalculationMethod
from egret.data.model_data import zip_items
from egret.parsers.matpower_parser import create_ModelData
import egret.data.data_utils_deprecated as data_utils_deprecated
import egret.model_library.decl as decl
import egret.common.lazy_ptdf_utils as lpu


def _include_system_feasibility_slack(model, gen_attrs, bus_p_loads, bus_q_loads, penalty=10):
    import egret.model_library.decl as decl
    slack_init = 0
    slack_bounds = (0, sum(bus_p_loads.values()))
    decl.declare_var('p_slack_pos', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    slack_bounds = (0, sum(bus_q_loads.values()))
    decl.declare_var('q_slack_pos', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    p_rhs_kwargs = {'include_feasibility_slack_pos': 'p_slack_pos', 'include_feasibility_slack_neg': 'p_slack_neg'}
    q_rhs_kwargs = {'include_feasibility_slack_pos': 'q_slack_pos', 'include_feasibility_slack_neg': 'q_slack_neg'}

    p_penalty = penalty * (max([gen_attrs['p_cost'][k]['values'][1] for k in gen_attrs['names']]) + 1)
    q_penalty = penalty * (max(gen_attrs.get('q_cost', gen_attrs['p_cost'])[k]['values'][1] for k in gen_attrs['names']) + 1)

    model._p_penalty = p_penalty
    model._q_penalty = q_penalty

    penalty_expr = get_balance_penalty_expr(model)

    return p_rhs_kwargs, q_rhs_kwargs, penalty_expr

def get_balance_penalty_expr(model):

    p_penalty = model._p_penalty
    q_penalty = model._q_penalty
    penalty_expr = p_penalty * (model.p_slack_pos + model.p_slack_neg) \
                   + q_penalty * (model.q_slack_pos + model.q_slack_neg)

    return penalty_expr

def _include_pf_feasibility_slack(model, branch_attrs, penalty=2000):
    import egret.model_library.decl as decl
    slack_init = {k: 0 for k in branch_attrs['names']}
    slack_bounds = {k: (0,inf) for k in branch_attrs['names']}
    decl.declare_var('pf_slack_pos', model=model, index_set=branch_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('pf_slack_neg', model=model, index_set=branch_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    pf_rhs_kwargs = {'include_feasibility_slack_pos': 'pf_slack_pos', 'include_feasibility_slack_neg': 'pf_slack_neg'}

    model._pf_penalty = penalty
    model._pf_rhs_kwargs = pf_rhs_kwargs

    penalty_expr = get_pf_penalty_expr(model, branch_attrs)

    return pf_rhs_kwargs, penalty_expr

def get_pf_penalty_expr(model, branch_attrs):

    penalty = model._pf_penalty

    penalty_expr = penalty * (sum(model.pf_slack_pos[k] + model.pf_slack_neg[k] for k in branch_attrs["names"]))

    return penalty_expr

def _include_qf_feasibility_slack(model, branch_attrs, penalty=2000):
    import egret.model_library.decl as decl
    slack_init = {k: 0 for k in branch_attrs['names']}
    slack_bounds = {k: (0,inf) for k in branch_attrs['names']}
    decl.declare_var('qf_slack_pos', model=model, index_set=branch_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('qf_slack_neg', model=model, index_set=branch_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    qf_rhs_kwargs = {'include_feasibility_slack_pos': 'qf_slack_pos', 'include_feasibility_slack_neg': 'qf_slack_neg'}

    model._qf_penalty = penalty
    model._qf_rhs_kwargs = qf_rhs_kwargs

    penalty_expr = get_qf_penalty_expr(model, branch_attrs)

    return qf_rhs_kwargs, penalty_expr

def get_qf_penalty_expr(model, branch_attrs):

    penalty = model._qf_penalty

    penalty_expr = penalty * (sum(model.qf_slack_pos[k] + model.qf_slack_neg[k] for k in branch_attrs["names"]))

    return penalty_expr

def _include_v_feasibility_slack(model, bus_attrs, gen_attrs, penalty=100):
    import egret.model_library.decl as decl
    slack_init = {k: 0 for k in bus_attrs['names']}
    slack_bounds = {k: (0,inf) for k in bus_attrs['names']}
    decl.declare_var('v_slack_pos', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('v_slack_neg', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    v_rhs_kwargs = {'include_feasibility_slack_pos': 'v_slack_pos', 'include_feasibility_slack_neg': 'v_slack_neg'}

    max_cost = max([gen_attrs['p_cost'][k]['values'][1] for k in gen_attrs['names']]) + 1

    model._v_penalty = penalty * max_cost
    model._v_rhs_kwargs = v_rhs_kwargs

    penalty_expr = get_v_penalty_expr(model, bus_attrs)

    return v_rhs_kwargs, penalty_expr

def get_v_penalty_expr(model, bus_attrs):

    penalty = model._v_penalty

    penalty_expr = penalty * (sum(model.v_slack_pos[k] + model.v_slack_neg[k] for k in bus_attrs["names"]))

    return penalty_expr

def create_fixed_fdf_model(model_data, **kwargs):
    ## creates an FDF model with fixed m.pg and m.qg, and relaxed power balance

    kwlist = list(kwargs.keys())
    if 'include_feasibility_slack' not in kwlist:
        kwargs['include_feasibility_slack'] = True
    if 'include_v_feasibility_slack' not in kwlist:
        kwargs['include_v_feasibility_slack']  = True
    model, md = create_fdf_model(model_data, **kwargs)

    gens = dict(model_data.elements(element_type='generator'))
    baseMVA = model_data.data['system']['baseMVA']

    for g, pg in model.pg.items():
        pg.value = gens[g]['pg'] / baseMVA
    for g, qg in model.qg.items():
        qg.value = gens[g]['qg'] / baseMVA

    model.pg.fix()
    model.qg.fix()

    return model, md


def create_fdf_model(model_data, include_feasibility_slack=False, include_v_feasibility_slack=False,
                     include_pf_feasibility_slack=False, include_qf_feasibility_slack=False,
                     ptdf_options=None, include_q_balance=False, calculation_method=SensitivityCalculationMethod.INVERT):

    if ptdf_options is None:
        ptdf_options = dict()

    lpu.populate_default_ptdf_options(ptdf_options)

    baseMVA = model_data.data['system']['baseMVA']
    lpu.check_and_scale_ptdf_options(ptdf_options, baseMVA)

    # model_data.return_in_service()
    # md = model_data
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    data_utils_deprecated.create_dicts_of_fdf(md)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    load_attrs = md.attributes(element_type='load')
    shunt_attrs = md.attributes(element_type='shunt')
    system_attrs = md.data['system']

    inlet_branches_by_bus, outlet_branches_by_bus = tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    buses_with_gens = tx_utils.buses_with_gens(gens)

    model = pe.ConcreteModel()

    model._ptdf_options = ptdf_options

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, bus_attrs['names'], initialize=bus_p_loads)
    libbus.declare_var_ql(model, bus_attrs['names'], initialize=bus_q_loads)
    model.pl.fix()
    model.ql.fix()

    ### declare the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the polar voltages
    libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )

    ### include the feasibility slack for the bus balances
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_system_feasibility_slack(model, gen_attrs, bus_p_loads, bus_q_loads)

    pf_rhs_kwargs = {}
    qf_rhs_kwargs = {}
    v_rhs_kwargs = {}
    if include_pf_feasibility_slack:
        pf_rhs_kwargs, pf_penalty_expr = _include_pf_feasibility_slack(model, branch_attrs)
    if include_qf_feasibility_slack:
        qf_rhs_kwargs, qf_penalty_expr = _include_qf_feasibility_slack(model, branch_attrs)
    if include_v_feasibility_slack:
        v_rhs_kwargs, v_penalty_expr = _include_v_feasibility_slack(model, bus_attrs, gen_attrs)

    ### declare the generator real and reactive power
    pg_init = {k: gens[k]['pg'] for k in gens.keys()}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=pg_init,
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )

    qg_init = {k: gens[k]['qg'] for k in gens.keys()}
    libgen.declare_var_qg(model, gen_attrs['names'], initialize=qg_init,
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )

    q_pos_bounds = {k: (0, inf) for k in gen_attrs['qg']}
    decl.declare_var('q_pos', model=model, index_set=gen_attrs['names'], bounds=q_pos_bounds)

    q_neg_bounds = {k: (0, inf) for k in gen_attrs['qg']}
    decl.declare_var('q_neg', model=model, index_set=gen_attrs['names'], bounds=q_neg_bounds)

    ### declare the net withdrawal variables and definition constraints
    p_net_withdrawal_init = {k: 0 for k in bus_attrs['names']}
    libbus.declare_var_p_nw(model, bus_attrs['names'], initialize=p_net_withdrawal_init)

    q_net_withdrawal_init = {k: 0 for k in bus_attrs['names']}
    libbus.declare_var_q_nw(model, bus_attrs['names'], initialize=q_net_withdrawal_init)

    libbus.declare_eq_p_net_withdraw_fdf(model,bus_attrs['names'],buses,bus_p_loads,gens_by_bus,bus_gs_fixed_shunts)

    libbus.declare_eq_q_net_withdraw_fdf(model,bus_attrs['names'],buses,bus_q_loads,gens_by_bus,bus_bs_fixed_shunts)


    ### declare the power flows in the branches
    s_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    s_lbub = dict()
    for k in branches.keys():
        if s_max[k] is None:
            s_lbub[k] = (None, None)
        else:
            s_lbub[k] = (-s_max[k],s_max[k])
    pf_bounds = s_lbub
    qf_bounds = s_lbub
    pfl_bounds = s_lbub
    qfl_bounds = s_lbub
    pf_init = {k: (branches[k]['pf'] - branches[k]['pt']) / 2 for k in branches.keys()}
    qf_init = {k: (branches[k]['qf'] - branches[k]['qt']) / 2 for k in branches.keys()}
    pfl_init = {k: branches[k]['pfl'] for k in branches.keys()}
    qfl_init = {k: branches[k]['qfl'] for k in branches.keys()}

    libbranch.declare_var_pfl(model=model,
                              index_set=branch_attrs['names'],
                              initialize=pfl_init)  # ,
    #                             bounds=pfl_bounds
    #                             )
    libbranch.declare_var_qfl(model=model,
                              index_set=branch_attrs['names'],
                              initialize=qfl_init)  # ,
    #                          bounds=qfl_bounds
    #                          )

    libbranch.declare_var_ploss(model=model, initialize=0)
    libbranch.declare_var_qloss(model=model, initialize=0)

    if ptdf_options['lazy']:

        monitor_init = set()
        for branch_name, branch in branches.items():
            pf = pf_init[branch_name]
            qf = qf_init[branch_name]
            lim = s_max[branch_name]
            if lim is not None:
                abs_slack = sqrt(lim**2 - (pf**2 + qf**2))
                rel_slack =  abs_slack / lim

                if abs_slack < ptdf_options['abs_thermal_init_tol'] or rel_slack < ptdf_options['rel_thermal_init_tol']:
                    monitor_init.add(branch_name)

        libbranch.declare_var_pf(model=model,
                                 index_set=branch_attrs['names'],
                                 initialize = pf_init,
                                 bounds=pf_bounds
                                 )
        libbranch.declare_var_qf(model=model,
                                 index_set=branch_attrs['names'],
                                 initialize = qf_init,
                                 bounds=qf_bounds
                                 )
        model.eq_pf_branch = pe.Constraint(branch_attrs['names'])
        model.eq_qf_branch = pe.Constraint(branch_attrs['names'])

        ## Note: constructor does not build constraints for index_set when 'lazy' is enabled
        libbranch.declare_fdf_thermal_limit(model=model,
                                            branches=branches,
                                            index_set=branch_attrs['names'],
                                            thermal_limits=s_max,
                                            cuts=1
                                            )

        ### add helpers for tracking monitored branches
        lpu.add_monitored_branch_tracker(model)
        thermal_idx_monitored = model._thermal_idx_monitored

        ## construct constraints of branches near limit
        ba_ptdf = branch_attrs['ptdf']
        ba_ptdf_c = branch_attrs['ptdf_c']
        ba_qtdf = branch_attrs['qtdf']
        ba_qtdf_c = branch_attrs['qtdf_c']
        for i,bn in enumerate(branch_attrs['names']):
            if bn in monitor_init:
                ## add pf definition
                expr = libbranch.get_expr_branch_pf_fdf_approx(model, bn, ba_ptdf[bn], ba_ptdf_c[bn],
                                                               rel_tol=ptdf_options['rel_ptdf_tol'],
                                                               abs_tol=ptdf_options['abs_ptdf_tol'],
                                                               **pf_rhs_kwargs)
                model.eq_pf_branch[bn] = model.pf[bn] == expr
                ## add qf definition
                expr = libbranch.get_expr_branch_qf_fdf_approx(model, bn, ba_qtdf[bn], ba_qtdf_c[bn],
                                                               rel_tol=ptdf_options['rel_qtdf_tol'],
                                                               abs_tol=ptdf_options['abs_qtdf_tol'],
                                                               **qf_rhs_kwargs)
                model.eq_qf_branch[bn] = model.qf[bn] == expr
                ## add thermal limit
                thermal_limit = s_max[bn]
                branch = branches[bn]
                libbranch.add_constr_branch_thermal_limit(model, branch, bn, thermal_limit)
                thermal_idx_monitored.append(i)
        logger.critical('{} of {} thermal constraints added to initial monitored set.'.format(len(monitor_init), len(branch_attrs['names'])))

    else:

        libbranch.declare_var_pf(model=model,
                                 index_set=branch_attrs['names'],
                                 initialize=pf_init,
                                 bounds=pf_bounds
                                 )

        libbranch.declare_var_qf(model=model,
                                 index_set=branch_attrs['names'],
                                 initialize=qf_init,
                                 bounds=qf_bounds
                                 )

        libbranch.declare_eq_branch_pf_fdf_approx(model=model,
                                                  index_set=branch_attrs['names'],
                                                  sensitivity=branch_attrs['ptdf'],
                                                  constant=branch_attrs['ptdf_c'],
                                                  rel_tol=ptdf_options['rel_ptdf_tol'],
                                                  abs_tol=ptdf_options['abs_ptdf_tol'],
                                                  **pf_rhs_kwargs
                                                  )

        libbranch.declare_eq_branch_qf_fdf_approx(model=model,
                                                  index_set=branch_attrs['names'],
                                                  sensitivity=branch_attrs['qtdf'],
                                                  constant=branch_attrs['qtdf_c'],
                                                  rel_tol=ptdf_options['rel_qtdf_tol'],
                                                  abs_tol=ptdf_options['abs_qtdf_tol'],
                                                  **qf_rhs_kwargs
                                                  )

        libbranch.declare_fdf_thermal_limit(model=model,
                                            branches=branches,
                                            index_set=branch_attrs['names'],
                                            thermal_limits=s_max,
                                            cuts=1
                                            )

    if ptdf_options['lazy_voltage']:

        shunt_buses = set()
        for shunt_name, shunt in shunts.items():
            if shunt['shunt_type'] == 'fixed':
                bus_name = shunt['bus']
                shunt_buses.add(bus_name)

        monitor_init.clear()
        for bus_name, bus in buses.items():
            vm = bus['vm']
            v_max = bus['v_max']
            v_min = bus['v_min']
            abs_slack = min( abs(vm - v_min) , abs(v_max - vm) )
            rel_slack =  abs_slack / (v_max - v_min)
            if abs_slack < ptdf_options['abs_vm_init_tol'] or rel_slack < ptdf_options['rel_vm_init_tol']:
                logger.info('adding vm {}: {} <= {} <= {}'.format(bus_name, v_min, vm, v_max))
                monitor_init.add(bus_name)

        model.eq_vm_bus = pe.Constraint(bus_attrs['names'])

        lpu.add_monitored_vm_tracker(model)

        # add shunt and/or generator buses
        monitor_init = monitor_init.union(shunt_buses)
        monitor_init = monitor_init.union(buses_with_gens)

        for i,bn in enumerate(bus_attrs['names']):
            if bn in monitor_init:
                bus = buses[bn]
                vdf = bus['vdf']
                vdf_c = bus['vdf_c']
                expr = libbus.get_vm_expr_vdf_approx(model, bn, vdf, vdf_c,
                                        rel_tol=ptdf_options['rel_vdf_tol'],
                                        abs_tol=ptdf_options['abs_vdf_tol'],
                                        **v_rhs_kwargs)

                model.eq_vm_bus[bn] = model.vm[bn] == expr
                model._vm_idx_monitored.append(i)
        mon_message = '{} of {} voltage constraints added to initial monitored set'.format(len(monitor_init),
                                                                                    len(bus_attrs['names']))
        mon_message += ' ({} shunt devices)'.format(len(shunt_buses))
        mon_message += ' ({} generator buses)'.format(len(buses_with_gens))
        logger.critical(mon_message)

    else:
        libbus.declare_eq_vm_vdf_approx(model=model,
                                        index_set=bus_attrs['names'],
                                        sensitivity=bus_attrs['vdf'],
                                        constant=bus_attrs['vdf_c'],
                                        rel_tol=ptdf_options['rel_vdf_tol'],
                                        abs_tol=ptdf_options['abs_vdf_tol'],
                                        **v_rhs_kwargs
                                        )


    libbranch.declare_eq_branch_pfl_fdf_approx(model=model,
                                               index_set=branch_attrs['names'],
                                               sensitivity=branch_attrs['pldf'],
                                               constant=branch_attrs['pldf_c'],
                                               rel_tol=ptdf_options['rel_pldf_tol'],
                                               abs_tol=ptdf_options['abs_pldf_tol'],
                                               )

    libbranch.declare_eq_branch_qfl_fdf_approx(model=model,
                                               index_set=branch_attrs['names'],
                                               sensitivity=branch_attrs['qldf'],
                                               constant=branch_attrs['qldf_c'],
                                               rel_tol=ptdf_options['rel_qldf_tol'],
                                               abs_tol=ptdf_options['abs_qldf_tol'],
                                               )

    # residual loss functions
    libbranch.declare_eq_ploss_fdf_simplified(model=model,
                                           sensitivity=bus_attrs['ploss_resid_sens'],
                                           constant=system_attrs['ploss_resid_const'],
                                           rel_tol=ptdf_options['rel_ploss_tol'],
                                           abs_tol=ptdf_options['abs_ploss_tol'],
                                           )

    libbranch.declare_eq_qloss_fdf_simplified(model=model,
                                           sensitivity=bus_attrs['qloss_resid_sens'],
                                           constant=system_attrs['qloss_resid_const'],
                                           rel_tol=ptdf_options['rel_qloss_tol'],
                                           abs_tol=ptdf_options['abs_qloss_tol'],
                                           )


    ### declare the p balance
    libbus.declare_eq_p_balance_fdf(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_p_loads=bus_p_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                    include_losses=branch_attrs['names'],
                                    **p_rhs_kwargs
                                    )

    ### declare the q balance
    if include_q_balance:
        libbus.declare_eq_q_balance_fdf(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_q_loads=bus_q_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                    include_losses=branch_attrs['names'],
                                    **q_rhs_kwargs
                                    )
        ### fix the reference bus
        #ref_bus = md.data['system']['reference_bus']
        #model.vm[ref_bus].fix(buses[ref_bus]['vm'])

    libgen.declare_eq_q_fdf_deviation(model=model,
                                      index_set=gen_attrs['names'],
                                      gens=gens)

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_fdf_cost(model=model,
                                            index_set=gen_attrs['names'],
                                            p_costs=gen_attrs['p_cost']
                                            )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    obj_expr += sum(model.qg_operating_cost[gen_name] for gen_name in model.qg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr
    if include_v_feasibility_slack:
        obj_expr += v_penalty_expr
    if include_pf_feasibility_slack:
        obj_expr += pf_penalty_expr
    if include_qf_feasibility_slack:
        obj_expr += qf_penalty_expr
    model.obj = pe.Objective(expr=obj_expr)

    return model, md


def _load_solution_to_model_data(m, md, results):
    import numpy as np
    from pyomo.environ import value
    from egret.model_library.transmission.tx_utils import unscale_ModelData_to_pu
    from egret.model_library.transmission.tx_calc import linsolve_theta_fdf, linsolve_vmag_fdf

    duals = md.data['results']['duals']

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))

    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    buses_idx = bus_attrs['names']
    branches_idx = branch_attrs['names']
    mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(bus_attrs['names'])}

    # remove penalties from objective function
    penalty_cost = 0
    if hasattr(m,'_p_penalty') and hasattr(m,'_q_penalty'):
        penalty_cost += value(get_balance_penalty_expr(m))
    if hasattr(m,'_pf_penalty'):
        penalty_cost += value(get_pf_penalty_expr(m, branch_attrs))
    if hasattr(m,'_qf_penalty'):
        penalty_cost += value(get_qf_penalty_expr(m, branch_attrs))
    if hasattr(m,'_v_penalty'):
        penalty_cost += value(get_v_penalty_expr(m, bus_attrs))

    md.data['system']['total_cost'] = value(m.obj)
    md.data['system']['penalty_cost'] = penalty_cost
    md.data['system']['ploss'] = sum(value(m.pfl[b]) for b,b_dict in branches.items())
    md.data['system']['qloss'] = sum(value(m.qfl[b]) for b,b_dict in branches.items())

    ## back-solve for theta/vmag then solve for p/q power flows
    THETA = linsolve_theta_fdf(m, md, solve_sparse_system=True)
    Ft = md.data['system']['Ft']
    ft_c = md.data['system']['ft_c']
    PFV = Ft.dot(THETA) + ft_c

    VMAG = linsolve_vmag_fdf(m, md, solve_sparse_system=True)
    Fv = md.data['system']['Fv']
    fv_c = md.data['system']['fv_c']
    QFV = Fv.dot(VMAG) + fv_c

    ## initialize LMP energy components
    if duals:
        LMPE = value(m.dual[m.eq_p_balance])
        LMPL = np.zeros(len(buses_idx))
        LMPC = np.zeros(len(buses_idx))

        if hasattr(m,'eq_q_balance'):
            QLMPE = value(m.dual[m.eq_q_balance])
        else:
            QLMPE = 0
        QLMPL = np.zeros(len(buses_idx))
        QLMPC = np.zeros(len(buses_idx))

    # branch data
    for i, branch_name in enumerate(branches_idx):
        k_dict = branches[branch_name]
        k_dict['pf'] = PFV[i]
        k_dict['qf'] = QFV[i]
        if duals and branch_name in m.eq_pf_branch:
            PFD = value(m.dual[m.eq_pf_branch[branch_name]])
            QFD = value(m.dual[m.eq_qf_branch[branch_name]])
            PLD = value(m.dual[m.eq_pfl_branch[branch_name]])
            QLD = value(m.dual[m.eq_qfl_branch[branch_name]])
            k_dict['pf_dual'] = PFD
            k_dict['qf_dual'] = QFD
            k_dict['pfl_dual'] = PLD
            k_dict['qfl_dual'] = QLD
            if PFD != 0:
                ptdf = k_dict['ptdf']
                for j, bus_name in enumerate(buses_idx):
                    LMPC[j] += ptdf[bus_name] * PFD
            if QFD != 0:
                qtdf = k_dict['qtdf']
                for j, bus_name in enumerate(buses_idx):
                    QLMPC[j] += qtdf[bus_name] * QFD
            if PLD != 0:
                pldf = k_dict['pldf']
                for j, bus_name in enumerate(buses_idx):
                    LMPL[j] += pldf[bus_name] * PLD
            if QLD != 0:
                qldf = k_dict['qldf']
                for j, bus_name in enumerate(buses_idx):
                    QLMPL[j] += qldf[bus_name] * QLD

    # bus data
    for i,b in enumerate(buses_idx):
        b_dict = buses[b]
        if duals:
            b_dict['lmp'] = LMPE + LMPL[i] + LMPC[i]
            b_dict['qlmp'] = QLMPE + QLMPL[i] + QLMPC[i]
        b_dict['pl'] = value(m.pl[b])
        b_dict['ql'] = value(m.ql[b])
        b_dict['va'] = THETA[i]
        b_dict['vm'] = VMAG[i]

        tol = 1e-6
        if b_dict['vm'] < b_dict['v_min'] - tol or b_dict['vm'] > b_dict['v_max'] + tol:
            logger.warning('BAD vm DATA-------: bus {}, vm {}'.format(b, b_dict['vm']))

    ## generator data
    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    unscale_ModelData_to_pu(md, inplace=True)

    return


def solve_fdf(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                fdf_model_generator = create_fdf_model,
                return_model = False,
                return_results = False,
                **kwargs):
    '''
    Create and solve a new acopf model

    Parameters
    ----------
    model_data : egret.data.ModelData
        An egret ModelData object with the appropriate data loaded.
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
    fdf_model_generator : function (optional)
        Function for generating the fdf model. Default is
        egret.models.acopf.create_fdf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    import time
    from egret.common.solver_interface import _solve_model, _load_persistent
    from egret.common.lazy_ptdf_utils import _lazy_model_solve_loop, LazyPTDFTerminationCondition
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

    m, md = fdf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    ## flag for persistent solver
    persistent_solver = isinstance(solver, PersistentSolver) or 'persistent' in solver

    if persistent_solver:
        vars_to_load = list()
        vars_to_load.extend(m.p_nw.values())
        vars_to_load.extend(m.q_nw.values())
        vars_to_load.extend(m.pf.values())
        vars_to_load.extend(m.qf.values())
        vars_to_load.extend(m.vm.values())
    else:
        vars_to_load = None

    m, results, solver = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,options=options,return_solver=True)

    if persistent_solver or solver.name=='gurobi_direct':
        init_solve_time = results.Solver[0]['Wallclock time']
    else:
        init_solve_time = results.Solver.Time

    start_loop_time = time.time()
    lazy_solve_loop = fdf_model_generator == create_fdf_model \
                        and m._ptdf_options['lazy']
    if lazy_solve_loop:
        ## cache the results object from the first solve
        results_init = results

        iter_limit = m._ptdf_options['iteration_limit']
        term_cond, results, iterations = _lazy_model_solve_loop(m, md, solver, timelimit=timelimit, solver_tee=solver_tee,
                                           symbolic_solver_labels=symbolic_solver_labels,iteration_limit=iter_limit,
                                           vars_to_load = vars_to_load)

        ## in this case, either we're not using lazy or
        ## we never re-solved
        if results is None:
            results = results_init

    loop_time = time.time() - start_loop_time
    total_time = init_solve_time + loop_time

    ### Note this results has nothing to do with that solved on line 810 in the _lazy_model_solve_loop
    if not hasattr(md,'results'):
        md.data['results'] = dict()
    md.data['results']['time'] = total_time
    md.data['results']['#_cons'] = results.Problem[0]['Number of constraints']
    md.data['results']['#_vars'] = results.Problem[0]['Number of variables']
    md.data['results']['#_nz'] = results.Problem[0]['Number of nonzeros']
    md.data['results']['termination'] = results.solver.termination_condition.__str__()
    if lazy_solve_loop:
        md.data['results']['iterations'] = iterations

    if persistent_solver:
        _load_persistent(solver, m)
    ## if there's an issue loading dual values,
    ## the _load_persistent call above will remove
    ## the dual attribute in place of raising an Exception
    duals = hasattr(m, 'dual')
    md.data['results']['duals'] = duals


    if results.Solver.status.key == 'ok':
        _load_solution_to_model_data(m, md, results)

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md


def compare_fdf_options(md):

    import os
    from egret.parsers.matpower_parser import create_ModelData
    from egret.data.test_utils import repopulate_acpf_to_modeldata
    from egret.models.tests.test_approximations import create_new_model_data

    # set case and filepath
    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case14_ieee.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    md = create_ModelData(matpower_file)

    logger = logging.getLogger('egret')
    logger.setLevel(logging.ERROR)

    # initialize solution dicts
    pg_dict = dict()
    qg_dict = dict()
    pf_dict = dict()
    pfl_dict = dict()
    qf_dict = dict()
    qfl_dict = dict()
    va_dict = dict()
    vm_dict = dict()
    acpf_slack = dict()
    pf_error = dict()
    qf_error = dict()
    vm_viol_dict = dict()
    thermal_viol_dict = dict()

    def update_solution_dicts(md, name="model_name"):

        filename = md.data['system']['model_name'] + '_fdf_' + '{}'.format(name)
        md.data['system']['filename'] = filename

        try:
            print('...Solving ACPF for model {}.'.format(name))
            acpf_to_md(md)
        except Exception as e:
            return e

        gen = md.attributes(element_type='generator')
        bus = md.attributes(element_type='bus')
        branch = md.attributes(element_type='branch')
        acpf_data = md.data['acpf_data']

        pg_dict.update({name: gen['pg']})
        qg_dict.update({name: gen['qg']})
        pf_dict.update({name: branch['pf']})
        pfl_dict.update({name: branch['pfl']})
        qf_dict.update({name: branch['qf']})
        qfl_dict.update({name: branch['qfl']})
        va_dict.update({name: bus['va']})
        vm_dict.update({name: bus['vm']})
        acpf_slack.update({name: {'slack' : acpf_data['acpf_slack']}})
        pf_error.update({name: acpf_data['pf_error']})
        qf_error.update({name: acpf_data['qf_error']})
        vm_viol_dict.update({name: acpf_data['vm_viol']})
        thermal_viol_dict.update({name: acpf_data['thermal_viol']})

    def acpf_to_md(md):
        try:
            repopulate_acpf_to_modeldata(md, write_to_json=True)
        except Exception as e:
            return e
        logger.critical('ACPF was successful')


    # solve ACOPF
    print('Solve ACOPF....')
    from egret.models.acopf import solve_acopf
    md_ac, m_ac, results = solve_acopf(md, "ipopt", return_model=True, return_results=True, solver_tee=False)
    print('ACOPF cost: $%3.2f' % md_ac.data['system']['total_cost'])
    print(results.Solver)

    md_ac = create_new_model_data(md_ac, 1.0)
    compare={}

    #solve D-LOPF default
    print('Solve D-LOPF (lazy)....')
    kwargs = {}
    ptdf_options = {}
    ptdf_options['lazy'] = True
    ptdf_options['lazy_voltage'] = True
    #ptdf_options['rel_ptdf_tol'] = 1e-2
    #ptdf_options['rel_qtdf_tol'] = 1e-2
    #ptdf_options['rel_pldf_tol'] = 1e-2
    #ptdf_options['rel_qldf_tol'] = 1e-2
    #ptdf_options['rel_vdf_tol'] = 1e-2
    kwargs['ptdf_options'] = ptdf_options
    kwargs['include_v_feasibility_slack'] = False
    kwargs['include_pf_feasibility_slack'] = False
    kwargs['include_qf_feasibility_slack'] = False
    kwargs['include_feasibility_slack'] = False
    try:
        md = solve_fdf(md_ac, "gurobi_persistent", return_model=False, return_results=False, solver_tee=False, **kwargs)
        print('Lazy cost: $%3.2f' % md.data['system']['total_cost'])
        update_solution_dicts(md,'lazy')
        compare['lazy'] = md.data['results']
        compare['lazy']['cost'] = md.data['system']['total_cost']
        print(compare)
    except Exception as e:
        raise e
        message = str(e)
        print(message)


    #solve D-LOPF tolerances
    print('Solve D-LOPF (default)....')
    kwargs = {}
    ptdf_options = {}
    ptdf_options['lazy'] = False
    ptdf_options['lazy_voltage'] = False
    #ptdf_options['rel_ptdf_tol'] = 1e-2
    #ptdf_options['rel_qtdf_tol'] = 1e-2
    #ptdf_options['rel_pldf_tol'] = 1e-2
    #ptdf_options['rel_qldf_tol'] = 1e-2
    #ptdf_options['rel_vdf_tol'] = 1e-2
    kwargs['ptdf_options'] = ptdf_options
    kwargs['include_v_feasibility_slack'] = False
    kwargs['include_pf_feasibility_slack'] = False
    kwargs['include_qf_feasibility_slack'] = False
    kwargs['include_feasibility_slack'] = False
    try:
        md = solve_fdf(md_ac, "gurobi_persistent", return_model=False, return_results=False, solver_tee=False, **kwargs)
        print('Default cost: $%3.2f' % md.data['system']['total_cost'])
        update_solution_dicts(md,'default')
        compare['default'] = md.data['results']
        compare['default']['cost'] = md.data['system']['total_cost']
    except Exception as e:
        raise e
        message = str(e)
        print(message)

    print(pd.DataFrame(compare))
    return

    # display results in dataframes
    compare_dict = {'pg' : pg_dict,
                    'qg' : qg_dict,
                    'pf' : pf_dict,
                    'qf' : qf_dict,
                    'pfl' : pfl_dict,
                    'qfl' : qfl_dict,
                    'va' : va_dict,
                    'vm' : vm_dict,
                    'slack' : acpf_slack,
                    'pf_error' : pf_error,
                    'qf_error' : qf_error,
                    }

    for mv,results in compare_dict.items():
        print('-{}:'.format(mv))
        compare_results(results,'lazy', 'default', display_results=False)


def nominal_test(argv=None, tml=None):
    # case list
    if len(argv)==0:
        idl = [0]
    else:
        print(argv)
        idl = test.get_case_names(flag=argv)
    # test model list
    if tml is None:
        tml = ['dlopf_full', 'dlopf_e4', 'dlopf_e2', 'dlopf_lazy_full', 'dlopf_lazy_e4', 'dlopf_lazy_e2']
    # run cases
    for idx in idl:
        test.run_nominal_test(idx=idx, tml=tml)

def loop_test(argv=None, tml=None):
    # case list
    if len(argv)==0:
        idl = [0]
    else:
        idl = test.get_case_names(flag=argv)
    # test model list
    if tml is None:
        tml = ['dlopf_full', 'dlopf_e4', 'dlopf_e2', 'dlopf_lazy_full', 'dlopf_lazy_e4', 'dlopf_lazy_e2']
    # run cases
    for idx in idl:
        test.run_test_loop(idx=idx, tml=tml)

def quick_solve(argv=None):

    logger = logging.getLogger('egret')
    logger.setLevel(logging.CRITICAL)

    # case list
    if len(argv)==0:
        idl = [0]
    else:
        idl = test.get_case_names(flag=argv)

    # model listlist
    tml = ['dlopf_full', 'dlopf_e4', 'dlopf_e2', 'dlopf_lazy_full', 'dlopf_lazy_e4', 'dlopf_lazy_e2']
    relaxations = [None, 'include_v_feasibility_slack', 'include_qf_feasibility_slack', 'include_pf_feasibility_slack']

    # Outer loop: test cases
    for idx in idl:
        test_case = test.idx_to_test_case(idx)
        test_case = test_case[3:]
        md = create_ModelData(test_case)

        print('>>>>> BEGIN SOLVE: acopf <<<<<')
        md_basept = solve_acopf(md, solver='ipopt', solver_tee=False)
        logger.critical('\t COST = ${:,.2f}'.format(md_basept.data['system']['total_cost']))
        logger.critical('\t TIME = {:.5f} seconds'.format(md_basept.data['results']['time']))

        # Inner loop: test models
        for tm in tml:
            tm_dict = test.generate_test_model_dict([tm])[tm]
            print('>>>>> BEGIN SOLVE: {} <<<<<'.format(tm))
            solve_func = tm_dict['solve_func']
            solver = tm_dict['solver']
            kwargs = tm_dict['kwargs']

            # Apply progressive relaxations if initial solve is infeasible
            for r in relaxations:
                if r is not None:
                    kwargs[r] = True
                    logger.critical('...applying relaxation with {}'.format(r))
                try:
                    md_out = solve_func(md_basept, solver=solver, **kwargs)
                    logger.critical('\t COST = ${:,.2f}'.format(md_out.data['system']['total_cost']))
                    logger.critical('\t TIME = {:.5f} seconds'.format(md_out.data['results']['time']))
                    is_feasible = True
                except Exception as e:
                    is_feasible = False
                    model_error = str(e)
                    logger.critical('failed: {}'.format(model_error))
                # end loop if solve is successful
                if is_feasible:
                    break

            # create results object if all relaxations failed
            if not is_feasible:
                md_out = md_basept
                md_out.data['results'] = {}
                md_out.data['results']['termination'] = 'infeasible'
                md_out.data['results']['exception'] = model_error
            else:
                md_out.data['system']['mult'] = 1

            test.record_results(tm, md_out)

        test.create_testcase_directory(test_case)

if __name__ == '__main__':

    #tml = None
    tml = ['dlopf_lazy_e2']
    if len(sys.argv)<=2:
        nominal_test(sys.argv[1], tml=tml)
    elif sys.argv[2]=='0':
        nominal_test(sys.argv[1], tml=tml)
    elif sys.argv[2]=='1':
        loop_test(sys.argv[1], tml=tml)
    elif sys.argv[2]=='2':
        quick_solve(sys.argv[1])
    else:
        message = 'file usage: model.py <case> <option>\n'
        message+= '\t case    = last N characters of cases to run\n'
        message+= '\t option  = 0 to run nominal or 1 for full test loop'
        print(message)