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
from math import pi, radians, inf


def _include_system_feasibility_slack(model, bus_attrs, gen_attrs, bus_p_loads, bus_q_loads, penalty=1000):
    import egret.model_library.decl as decl
    slack_init = 0
    slack_bounds = (0, sum(bus_p_loads.values()))
    decl.declare_var('p_slack_pos', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    slack_bounds = (0, sum(bus_q_loads.values()))
    decl.declare_var('q_slack_pos', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=model, index_set=bus_attrs["names"],
                     initialize=slack_init, bounds=slack_bounds
                     )
    p_rhs_kwargs = {'include_feasibility_slack_pos': 'p_slack_pos', 'include_feasibility_slack_neg': 'p_slack_neg'}
    q_rhs_kwargs = {'include_feasibility_slack_pos': 'q_slack_pos', 'include_feasibility_slack_neg': 'q_slack_neg'}

    p_penalty = penalty * (max([gen_attrs['p_cost'][k]['values'][1] for k in gen_attrs['names']]) + 1)
    q_penalty = penalty * (max(gen_attrs.get('q_cost', gen_attrs['p_cost'])[k]['values'][1] for k in gen_attrs['names']) + 1)

    penalty_expr = p_penalty * (sum(model.p_slack_pos[k] + model.p_slack_neg[k] for k in bus_attrs["names"])) \
        + q_penalty * (sum(model.q_slack_pos[k] + model.q_slack_neg[k] for k in bus_attrs["names"]))

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


def create_fixed_lccm_model(model_data, **kwargs):
    ## creates an FDF model with fixed m.pg and m.qg, and relaxed power balance

    model, md = create_lccm_model(model_data, include_feasibility_slack=True, include_v_feasibility_slack=True, **kwargs)

    for g, pg in model.pg.items():
        pg.value = value(m_ac.pg[g])
    for g, qg in model.qg.items():
        qg.value = value(m_ac.qg[g])

    model.pg.fix()
    model.qg.fix()

    return model, md


def create_lccm_model(model_data, include_feasibility_slack=False, include_v_feasibility_slack=False, calculation_method=SensitivityCalculationMethod.INVERT):
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    data_utils_deprecated.create_dicts_of_lccm(md)

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

    inlet_branches_by_bus, outlet_branches_by_bus = tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

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
    libbus.declare_var_va(model, bus_attrs['names'], initialize=bus_attrs['va'])

    ### fix the reference bus
    ref_bus = md.data['system']['reference_bus']
    ref_angle = md.data['system']['reference_bus_angle']
    model.va[ref_bus].fix(radians(ref_angle))

    ### include the feasibility slack for the bus balances
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_system_feasibility_slack(model, bus_attrs, gen_attrs, bus_p_loads, bus_q_loads)

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

    ### declare the net withdrawal variables (for later use in defining constraints with efficient 'LinearExpression')
    p_net_withdrawal_init = {k: 0 for k in bus_attrs['names']}
    libbus.declare_var_p_nw(model, bus_attrs['names'], initialize=p_net_withdrawal_init)

    q_net_withdrawal_init = {k: 0 for k in bus_attrs['names']}
    libbus.declare_var_q_nw(model, bus_attrs['names'], initialize=q_net_withdrawal_init)

    ### declare the current flows in the branches
    vr_init = {k: bus_attrs['vm'][k] * pe.cos(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    vj_init = {k: bus_attrs['vm'][k] * pe.sin(bus_attrs['va'][k]) for k in bus_attrs['vm']}
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
    pf_init = dict()
    pfl_init = dict()
    qf_init = dict()
    qfl_init = dict()
    for branch_name, branch in branches.items():
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        y_matrix = tx_calc.calculate_y_matrix_from_branch(branch)
        ifr_init = tx_calc.calculate_ifr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        ifj_init = tx_calc.calculate_ifj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itr_init = tx_calc.calculate_itr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itj_init = tx_calc.calculate_itj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        #TODO: this calculation seems redundant since line flows should already be in the model
        pf_init[branch_name] = (tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                -tx_calc.calculate_p(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))/2
        pfl_init[branch_name] = (tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                +tx_calc.calculate_p(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))
        qf_init[branch_name] = (tx_calc.calculate_q(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                -tx_calc.calculate_q(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))/2
        qfl_init[branch_name] = (tx_calc.calculate_q(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                +tx_calc.calculate_q(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))

    libbranch.declare_var_pf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pf_init)#,
#                             bounds=pf_bounds
#                             )
    libbranch.declare_var_pfl(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pfl_init)#,
#                             bounds=pfl_bounds
#                             )
    libbranch.declare_var_qf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=qf_init)#,
#                             bounds=qf_bounds
#                             )
    libbranch.declare_var_qfl(model=model,
                              index_set=branch_attrs['names'],
                              initialize=qfl_init)  # ,
#                              bounds=qfl_bounds
#                              )

    ### declare net withdrawal definition constraints
    libbus.declare_eq_p_net_withdraw_fdf(model,bus_attrs['names'],buses,bus_p_loads,gens_by_bus,bus_gs_fixed_shunts)
    libbus.declare_eq_q_net_withdraw_fdf(model,bus_attrs['names'],buses,bus_q_loads,gens_by_bus,bus_bs_fixed_shunts)

    libbranch.declare_eq_branch_pf_lccm_approx(model=model,
                                               index_set=branch_attrs['names'],
                                               sensitivity=branch_attrs['Ft'],
                                               constant=branch_attrs['ft_c'],
                                               rel_tol=0.,
                                               abs_tol=0.
                                               )

    libbranch.declare_eq_branch_qf_lccm_approx(model=model,
                                               index_set=branch_attrs['names'],
                                               sensitivity=branch_attrs['Fv'],
                                               constant=branch_attrs['fv_c'],
                                               rel_tol=0.,
                                               abs_tol=0.
                                               )

    libbranch.declare_eq_branch_pfl_lccm_approx(model=model,
                                                 index_set=branch_attrs['names'],
                                                 sensitivity=branch_attrs['Lt'],
                                                 constant=branch_attrs['lt_c'],
                                                 rel_tol=0.,
                                                 abs_tol=0.
                                                 )

    libbranch.declare_eq_branch_qfl_lccm_approx(model=model,
                                                 index_set=branch_attrs['names'],
                                                 sensitivity=branch_attrs['Lv'],
                                                 constant=branch_attrs['lv_c'],
                                                 rel_tol=0.,
                                                 abs_tol=0.
                                                 )

    ### declare the p balance
    libbus.declare_eq_p_balance_lccm_approx(model=model,
                                            index_set=bus_attrs['names'],
                                            buses=buses,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            inlet_branches_by_bus=inlet_branches_by_bus,
                                            outlet_branches_by_bus=outlet_branches_by_bus,
                                            **p_rhs_kwargs
                                            )

    ### declare the q balance
    libbus.declare_eq_q_balance_lccm_approx(model=model,
                                            index_set=bus_attrs['names'],
                                            buses=buses,
                                            bus_q_loads=bus_q_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                            inlet_branches_by_bus=inlet_branches_by_bus,
                                            outlet_branches_by_bus=outlet_branches_by_bus,
                                            **q_rhs_kwargs
                                    )

    ### declare the real power flow limits
    libbranch.declare_fdf_thermal_limit(model=model,
                                        index_set=branch_attrs['names'],
                                        thermal_limits=s_max
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
    from egret.model_library.transmission.tx_utils import unscale_ModelData_to_pu

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))

    md.data['system']['total_cost'] = value(m.obj)

    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    for b,b_dict in buses.items():
        b_dict['pl'] = value(m.pl[b])
        b_dict['ql'] = value(m.ql[b])
        b_dict['vm'] = value(m.vm[b])
        b_dict['va'] = value(m.va[b])

        b_dict['lmp'] = value(m.dual[m.eq_p_balance[b]])
        b_dict['qlmp'] = value(m.dual[m.eq_q_balance[b]])

    for k, k_dict in branches.items():
        k_dict['pf'] = value(m.pf[k])
        k_dict['qf'] = value(m.qf[k])
        k_dict['pfl'] = value(m.pfl[k])
        k_dict['qfl'] = value(m.qfl[k])
        k_dict['pf_dual'] = value(m.dual[m.eq_pf_branch[k]])
        k_dict['qf_dual'] = value(m.dual[m.eq_qf_branch[k]])

    md.data['system']['ploss'] = sum(k_dict['pfl'] for k, k_dict in branches.items())
    md.data['system']['qloss'] = sum(k_dict['qfl'] for k, k_dict in branches.items())

    unscale_ModelData_to_pu(md, inplace=True)

    return


def solve_lccm(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                lccm_model_generator = create_lccm_model,
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
    from egret.common.solver_interface import _solve_model

    m, md = lccm_model_generator(model_data, **kwargs)

    # for debugging purposes
    #m.pg.fix()
    #m.qg.fix()

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    m, results, solver = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,options=options,return_solver=True)

    if not hasattr(md,'results'):
        md.data['results'] = dict()
    print(results.Solver.items())
    md.data['results']['time'] = results.Solver.Time
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

def printresults(results):
    solver = results.attributes(element_type='Solver')

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData
    from pyomo.environ import value

    # set case and filepath
    path = os.path.dirname(__file__)
    #filename = 'pglib_opf_case3_lmbd.m'
    filename = 'pglib_opf_case5_pjm.m'
    #filename = 'pglib_opf_case14_ieee.m'
    #filename = 'pglib_opf_case30_ieee.m'
    #filename = 'pglib_opf_case57_ieee.m'
    #filename = 'pglib_opf_case118_ieee.m'
    #filename = 'pglib_opf_case162_ieee_dtc.m'
    #filename = 'pglib_opf_case179_goc.m'
    #filename = 'pglib_opf_case300_ieee.m'
    #filename = 'pglib_opf_case500_tamu.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf-master/', filename)
    md = create_ModelData(matpower_file)

    # keyword arguments
    kwargs = {'include_v_feasibility_slack':True}

    # solve ACOPF
    from egret.models.acopf import solve_acopf
    md_ac, m_ac, results = solve_acopf(md, "ipopt", return_model=True,return_results=True,solver_tee=False)
    print('ACOPF cost: $%3.2f' % md_ac.data['system']['total_cost'])
    print(results.Solver)
    gen = md_ac.attributes(element_type='generator')
    bus = md_ac.attributes(element_type='bus')
    branch = md_ac.attributes(element_type='branch')
    pg_dict = {'acopf' : gen['pg']}
    qg_dict = {'acopf' : gen['qg']}
    pt_dict = {'acopf' : branch['pt']}
    pf_dict = {'acopf' : branch['pf']}
    pfl_dict = {'acopf' : branch['pfl']}
    qt_dict = {'acopf' : branch['qt']}
    qf_dict = {'acopf' : branch['qf']}
    qfl_dict = {'acopf' : branch['qfl']}
    va_dict = {'acopf' : bus['va']}
    vm_dict = {'acopf' : bus['vm']}

    # solve D-LOPF
    kwargs={}
    options={}
    options['method'] = 1
    ptdf_options = {}
    ptdf_options['lazy'] = True
    ptdf_options['lazy_voltage'] = True
    ptdf_options['abs_ptdf_tol'] = 1e-6
    ptdf_options['abs_qtdf_tol'] = 5e-6
    ptdf_options['rel_vdf_tol'] = 10e-6
    kwargs['ptdf_options'] = ptdf_options
    from egret.models.fdf import solve_fdf
    md_dl, m_dl, results_dl = solve_fdf(md_ac, "gurobi_persistent", return_model=True,
                               return_results=True, solver_tee=False, options=options, **kwargs)

    print('DL-OPF cost: $%3.2f' % m_dl.obj.expr())
    print(results_dl.Solver)

    gen = md_dl.attributes(element_type='generator')
    bus = md_dl.attributes(element_type='bus')
    branch = md_dl.attributes(element_type='branch')
    pg_dict.update({'dlopf': gen['pg']})
    qg_dict.update({'dlopf': gen['qg']})
    pt_dict.update({'dlopf': branch['pt']})
    pf_dict.update({'dlopf': branch['pf']})
    pfl_dict.update({'dlopf': branch['pfl']})
    qt_dict.update({'dlopf': branch['qt']})
    qf_dict.update({'dlopf': branch['qf']})
    qfl_dict.update({'dlopf': branch['qfl']})
    va_dict.update({'dlopf': bus['va']})
    vm_dict.update({'dlopf': bus['vm']})

    # keyword arguments
    kwargs = {}
    #kwargs = {'include_v_feasibility_slack':True,'include_feasibility_slack':True}

    # solve S-LOPF
    md, m, results = solve_lccm(md_ac, "gurobi", lccm_model_generator=create_fixed_lccm_model, return_model=True,return_results=True,solver_tee=False, **kwargs)
    print('S-LOPF cost: $%3.2f' % md.data['system']['total_cost'])
    print(results.Solver)

    gen = md.attributes(element_type='generator')
    bus = md.attributes(element_type='bus')
    branch = md.attributes(element_type='branch')
    pg_dict.update({'slopf' : gen['pg']})
    qg_dict.update({'slopf' : gen['qg']})
    pt_dict.update({'slopf' : branch['pt']})
    pf_dict.update({'slopf' : branch['pf']})
    pfl_dict.update({'slopf' : branch['pfl']})
    qt_dict.update({'slopf' : branch['qt']})
    qf_dict.update({'slopf' : branch['qf']})
    qfl_dict.update({'slopf' : branch['qfl']})
    va_dict.update({'slopf' : bus['va']})
    vm_dict.update({'slopf' : bus['vm']})

    # display results in dataframes
    from egret.models.fdf import compare_results
    print('-pg:')
    compare_results(pg_dict,'slopf','dlopf')
#    print(pd.DataFrame(pg_dict))
    print('-qg:')
    compare_results(qg_dict,'slopf','dlopf')
#    print(pd.DataFrame(qg_dict))
#    print('-pt:')
#    print(pd.DataFrame(pt_dict))
    print('-pf:')
    compare_results(pf_dict,'slopf','dlopf')
#    print(pd.DataFrame(pf_dict))
    print('-pfl:')
    compare_results(pfl_dict,'slopf','dlopf')
#    print(pd.DataFrame(pfl_dict))
#    print('-qt:')
#    print(pd.DataFrame(qt_dict))
    print('-qf:')
    compare_results(qf_dict,'slopf','dlopf')
#    print(pd.DataFrame(qf_dict))
    print('-qfl:')
    compare_results(qfl_dict,'slopf','dlopf')
#    print(pd.DataFrame(qfl_dict))
#    print('-va:')
#    print(pd.DataFrame(va_dict))
    print('-vm:')
    compare_results(vm_dict,'slopf','dlopf')
#    print(pd.DataFrame(vm_dict))



# not solving pglib_opf_case57_ieee
# pglib_opf_case500_tamu
# pglib_opf_case162_ieee_dtc
# pglib_opf_case179_goc
# pglib_opf_case300_ieee
