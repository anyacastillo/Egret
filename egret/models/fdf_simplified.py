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
import pyomo.environ as pe
from math import inf, pi
import pandas as pd
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.branch_deprecated as libbranch_deprecated
import egret.model_library.transmission.gen as libgen
from egret.model_library.defn import ApproximationType, SensitivityCalculationMethod
from egret.data.model_data import zip_items
import egret.data.data_utils_deprecated as data_utils_deprecated
import egret.model_library.decl as decl
import egret.common.lazy_ptdf_utils as lpu


def _include_system_feasibility_slack(model, gen_attrs, bus_p_loads, bus_q_loads, penalty=1000):
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

    penalty_expr = p_penalty * (model.p_slack_pos + model.p_slack_neg) + q_penalty * (model.q_slack_pos + model.q_slack_neg)

    return p_rhs_kwargs, q_rhs_kwargs, penalty_expr


def _include_v_feasibility_slack(model, bus_attrs, penalty=100):
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

    penalty_expr = penalty * (sum(model.v_slack_pos[k] + model.v_slack_neg[k] for k in bus_attrs["names"]))

    return v_rhs_kwargs, penalty_expr


def create_fixed_fdf_model(model_data, **kwargs):
    ## creates an FDF model with fixed m.pg and m.qg, and relaxed power balance

    model, md = create_simplified_fdf_model(model_data, include_feasibility_slack=True, include_v_feasibility_slack=True, **kwargs)

    #c = 0

    for g, pg in model.pg.items():
        pg.value = value(m_ac.pg[g])
    for g, qg in model.qg.items():
        qg.value = value(m_ac.qg[g])
        #c += 1
        #if c==4:
        #    qg.fix()

    model.pg.fix()
    model.qg.fix()

    return model, md


def create_fixed_vm_fdf_model(model_data, **kwargs):
    ## creates an FDF model with fixed m.vm

    model, md = create_simplified_fdf_model(model_data, include_feasibility_slack=True, include_v_feasibility_slack=False, **kwargs)

    for b, vm in model.vm.items():
        vm.value = value(m_ac.vm[b])

    model.vm.fix()


def create_simplified_fdf_model(model_data, include_feasibility_slack=False, include_v_feasibility_slack=False,
                                ptdf_options=None, calculation_method=SensitivityCalculationMethod.INVERT):

    if ptdf_options is None:
        ptdf_options = dict()

    lpu.populate_default_ptdf_options(ptdf_options)
    ptdf_options['lazy'] = True
    ptdf_options['lazy_voltage'] = True

    baseMVA = model_data.data['system']['baseMVA']
    lpu.check_and_scale_ptdf_options(ptdf_options, baseMVA)

    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace=True)

    data_utils_deprecated.create_dicts_of_fdf_simplified(md)
    # TO BE DELETED: below and other functions called in create_dicts... method above
    # calculate_ptdf_ldf(branches, buses, index_set_branch, index_set_bus, reference_bus,
    #                   base_point=BasePointType.SOLUTION, sparse_index_set_branch=None, mapping_bus_to_idx=None)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))
    system_data = dict(md.data['system'])

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    load_attrs = md.attributes(element_type='load')
    shunt_attrs = md.attributes(element_type='shunt')
    system_attrs = md.data['system']

    inlet_branches_by_bus, outlet_branches_by_bus = tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

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
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_system_feasibility_slack(model, gen_attrs, bus_p_loads,
                                                                                     bus_q_loads)

    v_rhs_kwargs = {}
    if include_v_feasibility_slack:
        v_rhs_kwargs, v_penalty_expr = _include_v_feasibility_slack(model, bus_attrs)

    ### declare the generator real and reactive power
    pg_init = {k: (gen_attrs['p_min'][k] + gen_attrs['p_max'][k]) / 2.0 for k in gen_attrs['pg']}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=pg_init,
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )

    qg_init = {k: (gen_attrs['q_min'][k] + gen_attrs['q_max'][k]) / 2.0 for k in gen_attrs['qg']}
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

    libbus.declare_eq_p_net_withdraw_fdf(model, bus_attrs['names'], buses, bus_p_loads, gens_by_bus,
                                         bus_gs_fixed_shunts)
    libbus.declare_eq_q_net_withdraw_fdf(model, bus_attrs['names'], buses, bus_q_loads, gens_by_bus,
                                         bus_bs_fixed_shunts)

    ### declare the current flows in the branches #TODO: Why are we calculating currents for FDF initialization? Only need P,Q,V,theta
    s_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    s_lbub = dict()
    for k in branches.keys():
        if s_max[k] is None:
            s_lbub[k] = (None, None)
        else:
            s_lbub[k] = (-s_max[k], s_max[k])
    pf_bounds = s_lbub
    qf_bounds = s_lbub
    pfl_bounds = s_lbub
    qfl_bounds = s_lbub
    pf_init = dict()
    qf_init = dict()
    _len_branch = len(branch_attrs['names'])
    ploss_init = sum(branches[bn]['pf'] + branches[bn]['pt'] for bn in branch_attrs['names']) / baseMVA
    qloss_init = sum(branches[bn]['qf'] + branches[bn]['qt'] for bn in branch_attrs['names']) / baseMVA
    #ploss_init = 0
    #qloss_init = 0


    ### declare the branch power flow variables and approximation constraints
    if ptdf_options['lazy']:

        for branch_name, branch in branches.items():
            pf_init[branch_name] = 0
            qf_init[branch_name] = 0

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

        libbranch.declare_fdf_thermal_limit(model=model,
                                            index_set=branch_attrs['names'],
                                            thermal_limits=s_max,
                                            )

        ### add helpers for tracking monitored branches
        lpu.add_monitored_branch_tracker(model)

    else:

        for branch_name, branch in branches.items():
            pf_init[branch_name] = (branch['pf'] - branch['pt']) / 2
            qf_init[branch_name] = (branch['qf'] - branch['qt']) / 2

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
                                                  rel_tol=None,
                                                  abs_tol=None
                                                  )

        libbranch.declare_eq_branch_qf_fdf_approx(model=model,
                                                  index_set=branch_attrs['names'],
                                                  sensitivity=branch_attrs['qtdf'],
                                                  constant=branch_attrs['qtdf_c'],
                                                  rel_tol=None,
                                                  abs_tol=None
                                                  )

        libbranch.declare_fdf_thermal_limit(model=model,
                                            index_set=branch_attrs['names'],
                                            thermal_limits=s_max,
                                            )

    if ptdf_options['lazy_voltage']:

        shunt_buses = set()
        for shunt_name, shunt in shunts.items():
            if shunt['shunt_type'] == 'fixed':
                bus_name = shunt['bus']
                shunt_buses.add(bus_name)

        model.eq_vm_bus = pe.Constraint(bus_attrs['names'])

        lpu.add_monitored_vm_tracker(model)

        for i,bn in enumerate(bus_attrs['names']):
            if bn in shunt_buses:
                bus = buses[bn]
                vdf = bus['vdf']
                vdf_c = bus['vdf_c']
                expr = libbus.get_vm_expr_vdf_approx(model, bn, vdf, vdf_c)
                model.eq_vm_bus[bn] = model.vm[bn] == expr
                model._vm_idx_monitored.append(i)

    else:
        libbus.declare_eq_vm_vdf_approx(model=model,
                                        index_set=bus_attrs['names'],
                                        sensitivity=bus_attrs['vdf'],
                                        constant=bus_attrs['vdf_c'],
                                        rel_tol=None,
                                        abs_tol=None
                                        )


    libbranch.declare_var_ploss(model=model,
                              initialize=ploss_init)  # ,
    #                             bounds=pfl_bounds
    #                             )
    libbranch.declare_var_qloss(model=model,
                              initialize=qloss_init)  # ,
    #                              bounds=qfl_bounds
    #                              )

    libbranch.declare_eq_ploss_fdf_simplified(model=model,
                                           sensitivity=bus_attrs['ploss_sens'],
                                           constant=system_attrs['ploss_const'],
                                           rel_tol=None,
                                           abs_tol=None
                                           )

    libbranch.declare_eq_qloss_fdf_simplified(model=model,
                                           sensitivity=bus_attrs['qloss_sens'],
                                           constant=system_attrs['qloss_const'],
                                           rel_tol=None,
                                           abs_tol=None
                                           )


    ### declare the p balance
    libbus.declare_eq_p_balance_fdf_simplified(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_p_loads=bus_p_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                    **p_rhs_kwargs
                                    )

    ### declare the q balance
    libbus.declare_eq_q_balance_fdf_simplified(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_q_loads=bus_q_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                    **q_rhs_kwargs
                                    )

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
    model.obj = pe.Objective(expr=obj_expr)

    return model, md



def _load_solution_to_model_data(m, md, results):
    from pyomo.environ import value
    import numpy as np
    from egret.model_library.transmission.tx_utils import unscale_ModelData_to_pu
    from egret.model_library.transmission.tx_calc import linsolve_theta_fdf, linsolve_vmag_fdf

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))

    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    buses_idx = bus_attrs['names']
    branches_idx = branch_attrs['names']
    mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(buses_idx)}

    md.data['system']['total_cost'] = value(m.obj)
    md.data['system']['ploss'] = value(m.ploss)
    md.data['system']['qloss'] = value(m.qloss)

    ## back-solve for theta/vmag then solve for p/q power flows
    THETA = linsolve_theta_fdf(m, md)
    Ft = md.data['system']['Ft']
    ft_c = md.data['system']['ft_c']
    PFV = Ft.dot(THETA) + ft_c

    VMAG = linsolve_vmag_fdf(m, md)
    Fv = md.data['system']['Fv']
    fv_c = md.data['system']['fv_c']
    QFV = Fv.dot(VMAG) + fv_c

    ## initialize LMP energy components
    LMPE = value(m.dual[m.eq_p_balance])
    LMPL = np.zeros(len(buses_idx))
    LMPC = np.zeros(len(buses_idx))

    QLMPE = value(m.dual[m.eq_q_balance])
    QLMPL = np.zeros(len(buses_idx))
    QLMPC = np.zeros(len(buses_idx))

    # branch data
    for i, branch_name in enumerate(branches_idx):
        k_dict = branches[branch_name]
        k_dict['pf'] = PFV[i]
        k_dict['qf'] = QFV[i]
        if branch_name in m.eq_pf_branch:
            PFD = value(m.dual[m.eq_pf_branch[branch_name]])
            QFD = value(m.dual[m.eq_qf_branch[branch_name]])
            k_dict['pf_dual'] = PFD
            k_dict['qf_dual'] = QFD
            if PFD != 0:
                ptdf = k_dict['ptdf']
                for j, bus_name in enumerate(buses_idx):
                    LMPC[j] += ptdf[bus_name] * PFD
            if QFD != 0:
                qtdf = k_dict['qtdf']
                for j, bus_name in enumerate(buses_idx):
                    QLMPC[j] += qtdf[bus_name] * QFD

    # bus data
    for i,b in enumerate(buses_idx):
        PLD = value(m.dual[m.eq_ploss])
        QLD = value(m.dual[m.eq_qloss])
        b_dict = buses[b]
        LMPL[i] = PLD * b_dict['ploss_sens']
        QLMPL[i] = QLD * b_dict['ploss_sens']
        b_dict['lmp'] = LMPE + LMPL[i] + LMPC[i]
        b_dict['qlmp'] = QLMPE + QLMPL[i] + QLMPC[i]
        b_dict['pl'] = value(m.pl[b])
        b_dict['ql'] = value(m.ql[b])
        b_dict['va'] = THETA[i]
        b_dict['vm'] = VMAG[i]

    ## generator data
    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    unscale_ModelData_to_pu(md, inplace=True)

    return


def solve_fdf_simplified(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                fdf_model_generator = create_simplified_fdf_model,
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
        egret.models.fdf.create_simplified_fdf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    from egret.common.solver_interface import _solve_model
    from egret.common.lazy_ptdf_utils import _lazy_model_solve_loop
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

    m, md = fdf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    if isinstance(solver, PersistentSolver) or 'persistent' in solver:
        vars_to_load = [var for var in m.p_nw.values()]
        vars_to_load += [var for var in m.q_nw.values()]
    else:
        vars_to_load = None
    m, results, solver = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,options=options,return_solver=True,
                                      vars_to_load=vars_to_load)


    if fdf_model_generator == create_simplified_fdf_model and m._ptdf_options['lazy']:
        iter_limit = m._ptdf_options['iteration_limit']
        term_cond = _lazy_model_solve_loop(m, md, solver, timelimit=timelimit, solver_tee=solver_tee,
                                           symbolic_solver_labels=symbolic_solver_labels,iteration_limit=iter_limit,
                                           vars_to_load = vars_to_load)
    if isinstance(solver, PersistentSolver):
        solver.load_vars()


    if not hasattr(md,'results'):
        md.data['results'] = dict()
    # TODO: needs "manual" time recording due to use of persistent solver
    #md.data['results']['time'] = results.Solver.Time
    md.data['results']['#_cons'] = results.Problem[0]['Number of constraints']
    md.data['results']['#_vars'] = results.Problem[0]['Number of variables']
    md.data['results']['#_nz'] = results.Problem[0]['Number of nonzeros']
    md.data['results']['termination'] = results.solver.termination_condition.__str__()

    if results.Solver.status.key == 'ok':
        _load_solution_to_model_data(m, md, results)
        #m.vm.pprint()
        #m.v_slack_pos.pprint()
        #m.v_slack_neg.pprint()

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md

def compare_results(results, c1, c2, tol=1e-6):
    import numpy as np
    c1_results = results.get(c1)
    c2_results = results.get(c2)
    c1_array = np.fromiter(c1_results.values(), dtype=float)
    c2_array = np.fromiter(c2_results.values(), dtype=float)
    diff = (c1_array - c2_array)
    adiff = np.absolute(diff)
    idx = adiff.argmax()
    suma = sum(adiff)
    if suma < tol:
        print('Sum of absolute differences is less than {}.'.format(tol))
    else:
        print('Sum of absolute differences is {}.'.format(suma))
        out_str = 'Largest difference is {} at index {}'.format(diff[idx],idx+1)
        out_str += ' ({0} in {1} and {2} in {3}).'.format(c1_array[idx],c1,c2_array[idx],c2)
        print(out_str)


def printresults(results):
    solver = results.attributes(element_type='Solver')

def compare_to_acopf():
    import os
    from egret.parsers.matpower_parser import create_ModelData
    from pyomo.environ import value

    # set case and filepath
    path = os.path.dirname(__file__)
    #filename = 'pglib_opf_case3_lmbd.m'
    #filename = 'pglib_opf_case5_pjm.m'
    #filename = 'pglib_opf_case14_ieee.m'
    #filename = 'pglib_opf_case30_ieee.m'
    #filename = 'pglib_opf_case57_ieee.m'
    #filename = 'pglib_opf_case118_ieee.m'
    #filename = 'pglib_opf_case162_ieee_dtc.m'
    #filename = 'pglib_opf_case179_goc.m'
    filename = 'pglib_opf_case300_ieee.m'
    #filename = 'pglib_opf_case500_tamu.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    md = create_ModelData(matpower_file)

    # keyword arguments
    #kwargs = {'include_v_feasibility_slack': True}

    # solve ACOPF
    print('begin ACOPF...')
    from egret.models.acopf import solve_acopf
    md_ac, m_ac, results = solve_acopf(md, "ipopt", return_model=True, return_results=True, solver_tee=False)
    print('ACOPF cost: $%3.2f' % md_ac.data['system']['total_cost'])
    print(results.Solver)
    gen = md_ac.attributes(element_type='generator')
    bus = md_ac.attributes(element_type='bus')
    branch = md_ac.attributes(element_type='branch')
    system = md_ac.data['system']
    pg_dict = {'acopf': gen['pg']}
    qg_dict = {'acopf': gen['qg']}
    tmp_pf = branch['pf']
    tmp_pt = branch['pt']
    tmp = {key: (tmp_pf[key] - tmp_pt.get(key, 0)) / 2 for key in tmp_pf.keys()}
    pf_dict = {'acopf': tmp}
    tmp_qf = branch['qf']
    tmp_qt = branch['qt']
    tmp = {key: (tmp_qf[key] - tmp_qt.get(key, 0)) / 2 for key in tmp_qf.keys()}
    qf_dict = {'acopf': tmp}
    ploss_dict = {'acopf': system['ploss']}
    qloss_dict = {'acopf': system['qloss']}
    va_dict = {'acopf': bus['va']}
    vm_dict = {'acopf': bus['vm']}
    lmp_dict = {'acopf' : bus['lmp']}
    qlmp_dict = {'acopf' : bus['qlmp']}
    pf_dual = {'acopf' : branch['pf_dual']}
    qf_dual = {'acopf' : branch['qf_dual']}

    # keyword arguments
    kwargs = {}
    # kwargs = {'include_v_feasibility_slack':True,'include_feasibility_slack':True}

    # solve (fixed) FDF
    print('begin FDF...')
    md, m, results = solve_fdf_simplified(md_ac, "gurobi_persistent", fdf_model_generator=create_simplified_fdf_model,
                                          return_model=True, return_results=True, solver_tee=False, **kwargs)
    print('FDF cost: $%3.2f' % md.data['system']['total_cost'])
    print(results.Solver)
    if 'm.p_slack_pos' in locals():
        if value(m.p_slack_pos + m.p_slack_neg) > 1e-6:
            print('REAL POWER IMBALANCE: {}'.format(value(m.p_slack_pos + m.p_slack_neg)))
        if value(m.q_slack_pos + m.q_slack_neg) > 1e-6:
            print('REACTIVE POWER IMBALANCE: {}'.format(value(m.q_slack_pos + m.q_slack_neg)))
    gen = md.attributes(element_type='generator')
    bus = md.attributes(element_type='bus')
    branch = md.attributes(element_type='branch')
    system = md.data['system']
    pg_dict.update({'fdf': gen['pg']})
    qg_dict.update({'fdf': gen['qg']})
    pf_dict.update({'fdf': branch['pf']})
    qf_dict.update({'fdf': branch['qf']})
    ploss_dict.update({'fdf': system['ploss']})
    qloss_dict.update({'fdf': system['qloss']})
    va_dict.update({'fdf': bus['va']})
    vm_dict.update({'fdf': bus['vm']})
    lmp_dict.update({'fdf' : bus['lmp']})
    qlmp_dict.update({'fdf' : bus['qlmp']})
    pf_dual.update({'fdf' : branch['pf_dual']})
    qf_dual.update({'fdf' : branch['qf_dual']})

    # solve LCCM
    print('begin LCCM...')
    from egret.models.lccm import solve_lccm
    md, m, results = solve_lccm(md_ac, "gurobi", return_model=True, return_results=True, solver_tee=False, **kwargs)
    print('LCCM cost: $%3.2f' % md.data['system']['total_cost'])
    print(results.Solver)
    if 'm.p_slack_pos' in locals():
        if value(m.p_slack_pos + m.p_slack_neg) > 1e-6:
            print('REAL POWER IMBALANCE: {}'.format(value(m.p_slack_pos + m.p_slack_neg)))
        if value(m.q_slack_pos + m.q_slack_neg) > 1e-6:
            print('REACTIVE POWER IMBALANCE: {}'.format(value(m.q_slack_pos + m.q_slack_neg)))
    gen = md.attributes(element_type='generator')
    bus = md.attributes(element_type='bus')
    branch = md.attributes(element_type='branch')
    system = md.data['system']
    pg_dict.update({'lccm': gen['pg']})
    qg_dict.update({'lccm': gen['qg']})
    pf_dict.update({'lccm': branch['pf']})
    qf_dict.update({'lccm': branch['qf']})
    ploss_dict.update({'lccm': system['ploss']})
    qloss_dict.update({'lccm': system['qloss']})
    va_dict.update({'lccm': bus['va']})
    vm_dict.update({'lccm': bus['vm']})
    lmp_dict.update({'lccm' : bus['lmp']})
    qlmp_dict.update({'lccm' : bus['qlmp']})
    pf_dual.update({'lccm' : branch['pf_dual']})
    qf_dual.update({'lccm' : branch['qf_dual']})

    # display results in dataframes
    print('-pg:')
    #print(pd.DataFrame(pg_dict))
    compare_results(pg_dict, 'fdf', 'lccm')
    print('-qg:')
    #print(pd.DataFrame(qg_dict))
    compare_results(qg_dict, 'fdf', 'lccm')
    print('-pf:')
    #print(pd.DataFrame(pf_dict))
    compare_results(pf_dict, 'fdf', 'lccm')
    print('-qf:')
    #print(pd.DataFrame(qf_dict))
    compare_results(qf_dict, 'fdf', 'lccm')
    print('-ploss:')
    print(ploss_dict)
    #print(pd.DataFrame(ploss_dict[0]))
    print('-qloss:')
    print(qloss_dict)
    #print(pd.DataFrame(qloss_dict[0]))
    print('-va:')
    #print(pd.DataFrame(va_dict))
    compare_results(va_dict, 'fdf', 'lccm')
    print('-vm:')
    #print(pd.DataFrame(vm_dict))
    compare_results(vm_dict, 'fdf', 'lccm')
    print('-lmp:')
    #print(pd.DataFrame(lmp_dict))
    compare_results(lmp_dict, 'fdf', 'lccm')
    print('-qlmp:')
    #print(pd.DataFrame(qlmp_dict))
    compare_results(qlmp_dict, 'fdf', 'lccm')

if __name__ == '__main__':

    #compare_to_acopf()

    import os
    from egret.parsers.matpower_parser import create_ModelData
    from pyomo.environ import value

    # set case and filepath
    path = os.path.dirname(__file__)
    # filename = 'pglib_opf_case3_lmbd.m'
    # filename = 'pglib_opf_case5_pjm.m'
    # filename = 'pglib_opf_case14_ieee.m'
    # filename = 'pglib_opf_case30_ieee.m'
    # filename = 'pglib_opf_case57_ieee.m'
    # filename = 'pglib_opf_case118_ieee.m'
    # filename = 'pglib_opf_case162_ieee_dtc.m'
    # filename = 'pglib_opf_case179_goc.m'
    filename = 'pglib_opf_case300_ieee.m'
    # filename = 'pglib_opf_case500_tamu.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    md = create_ModelData(matpower_file)

    print('begin ACOPF...')
    from egret.models.acopf import solve_acopf
    md_ac, m_ac, results = solve_acopf(md, "ipopt", return_model=True, return_results=True, solver_tee=False)
    print('ACOPF cost: $%3.2f' % md_ac.data['system']['total_cost'])
    print(results.Solver)

    # keyword arguments
    kwargs = {}
    # kwargs = {'include_v_feasibility_slack':True,'include_feasibility_slack':True}

    print('begin FDF...')
    options={}
    options['method'] = 1
    md, m, results = solve_fdf_simplified(md_ac, "gurobi_persistent", fdf_model_generator=create_simplified_fdf_model,
                                          return_model=True, return_results=True, solver_tee=True
                                          , options=options, **kwargs)
    print('FDF cost: $%3.2f' % md.data['system']['total_cost'])
    print(results.Solver)

# not solving pglib_opf_case57_ieee
# pglib_opf_case500_tamu
# pglib_opf_case162_ieee_dtc
# pglib_opf_case179_goc
# pglib_opf_case300_ieee
