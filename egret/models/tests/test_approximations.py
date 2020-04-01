#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
 Tests linear OPF models against the ACOPF
 Usage:
    -python test_approximations --test_casesX
        Runs main() method

 Output functions:
    -solve_approximation_models(test_case, test_model_dict, init_min=0.9, init_max=1.1, steps=10)
        test_case: str, name of input data matpower file in pglib-opf-master
        test_model_dict: dict, list of models (key) and True/False (value) for which formulations/settings to
            run in the inner_loop_solves() method
        init_min: float (scalar), smallest demand multiplier to test approximation model. May be increased if
            ACOPF is infeasible
        init_max: float (scalar), largest demand multiplier to test approximation model. May be decreased if
            ACOPF is infeasible
        steps: integer, number of steps taken between init_min and init_max. Total of steps+1 model solves.
    -generate_sensitivity_plot(test_case,test_model_dict, data_generator=X)
        test_case: str, plots data from a test_case*.json wildcard search
        test_model_dict: dict, list of models (key) and True/False (value) for which formulations/settings to plot
        data_generator: function (perhaps from tx_utils.py) to pull data from JSON files
    -generate_pareto_plot(test_case, test_model_dict, y_axis_generator=Y, x_axis_generator=X, size_generator=None)
        test_case: str, plots data from a test_case*.json wildcard search
        test_model_dict: dict, list of models (key) and True/False (value) for which formulations/settings to plot
        y_axis_generator: function (perhaps from test_utils.py) to pull y-axis data from JSON files
        x_axis_generator: function (perhaps from test_utils.py) to pull x-axis data from JSON files
        size_generator: function (perhaps from test_utils.py) to pull dot size data from JSON files


 test_utils (tu) generator functions:
    solve_time
    num_constraints
    num_variables
    num_nonzeros
    model_density
    total_cost
    ploss
    qloss
    pgen
    qgen
    pflow
    qflow
    vmag
    acpf_slack
    sum_vm_UB_viol
    sum_vm_LB_viol
    sum_vm_viol
    sum_thermal_viol
    avg_vm_UB_viol
    avg_vm_LB_viol
    avg_vm_viol
    avg_thermal_viol
    max_vm_UB_viol
    max_vm_LB_viol
    max_vm_viol
    max_thermal_viol
    pct_vm_UB_viol
    pct_vm_LB_viol
    pct_vm_viol
    pct_thermal_viol


'''

import os, shutil, glob, json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as clrs
import matplotlib.cm as cmap
import seaborn as sns
from cycler import cycler
import math
import unittest
import logging
import egret.data.test_utils as tu
import egret.data.summary_plot_utils as spu
from pyomo.opt import SolverFactory, TerminationCondition
from egret.models.acopf import *
from egret.models.ccm import *
from egret.models.fdf import *
from egret.models.fdf_simplified import *
from egret.models.lccm import *
from egret.models.dcopf_losses import *
from egret.models.dcopf import *
from egret.models.copperplate_dispatch import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join

current_dir = os.path.dirname(os.path.abspath(__file__))
# test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]
case_names = ['pglib_opf_case3_lmbd',
              'pglib_opf_case5_pjm',
              'pglib_opf_case14_ieee',
              'pglib_opf_case24_ieee_rts',
              'pglib_opf_case30_as',
              'pglib_opf_case30_fsr',
              'pglib_opf_case30_ieee',
              'pglib_opf_case39_epri',
              'pglib_opf_case57_ieee',
              'pglib_opf_case73_ieee_rts',
              'pglib_opf_case89_pegase', ### not feasible at mult = 1.01 ###
              'pglib_opf_case118_ieee',
              'pglib_opf_case162_ieee_dtc',
              'pglib_opf_case179_goc',
              'pglib_opf_case200_tamu',
              'pglib_opf_case240_pserc',
              'pglib_opf_case300_ieee',
              'pglib_opf_case500_tamu',
              'pglib_opf_case588_sdet',
              'pglib_opf_case1354_pegase',
              'pglib_opf_case1888_rte',
              'pglib_opf_case1951_rte',
              'pglib_opf_case2000_tamu',
              'pglib_opf_case2316_sdet',
              'pglib_opf_case2383wp_k',
              'pglib_opf_case2736sp_k',
              'pglib_opf_case2737sop_k',
              'pglib_opf_case2746wop_k',
              'pglib_opf_case2746wp_k',
              'pglib_opf_case2848_rte',
              'pglib_opf_case2853_sdet',
              'pglib_opf_case2868_rte',
              'pglib_opf_case2869_pegase',
              'pglib_opf_case3012wp_k',
              'pglib_opf_case3120sp_k',
              'pglib_opf_case3375wp_k',
              'pglib_opf_case4661_sdet',
              'pglib_opf_case6468_rte',
              'pglib_opf_case6470_rte',
              'pglib_opf_case6495_rte',
              'pglib_opf_case6515_rte',
              'pglib_opf_case9241_pegase',
              'pglib_opf_case10000_tamu',
              'pglib_opf_case13659_pegase',
              ]
test_cases = [join('../../../download/pglib-opf-master/', f + '.m') for f in case_names]
#test_cases = [os.path.join(current_dir, 'download', 'pglib-opf-master', '{}.m'.format(i)) for i in case_names]

def get_case_names():
    return case_names


def set_acopf_basepoint_min_max(model_data, init_min=0.9, init_max=1.1, **kwargs):
    """
    returns AC basepoint solution and feasible min/max range
     - new min/max range b/c test case may not be feasible in [init_min to init_max]
    """
    md = model_data.clone_in_service()

    acopf_model = create_psv_acopf_model

    md_basept, m, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=True,
                                        return_results=True, solver_tee=False)

    # exit if base point does not return optimal
    if not results.solver.termination_condition == TerminationCondition.optimal:
        raise Exception('Base case acopf did not return optimal solution')

    # find feasible min and max demand multipliers
    else:
        mult_min = multiplier_loop(md, init=init_min, steps=10, acopf_model=acopf_model)
        mult_max = multiplier_loop(md, init=init_max, steps=10, acopf_model=acopf_model)

    return md_basept, mult_min, mult_max


def multiplier_loop(md, init=0.9, steps=10, acopf_model=create_psv_acopf_model):
    '''
    init < 1 searches for the lowest demand multiplier >= init that has an optimal acopf solution
    init > 1 searches for the highest demand multiplier <= init that has an optimal acopf solution
    steps determines the increments in [init, 1] where the search is made
    '''

    loads = dict(md.elements(element_type='load'))

    # step size
    inc = abs(1 - init) / steps

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # loop
    final_mult = None
    for step in range(0, steps):

        # for finding minimum
        if init < 1:
            mult = round(init - step * inc, 4)

        # for finding maximum
        elif init > 1:
            mult = round(init - step * inc, 4)

        # adjust load from init_min
        for k in loads.keys():
            loads[k]['p_load'] = init_p_loads[k] * mult
            loads[k]['q_load'] = init_q_loads[k] * mult

        try:
            md_, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=False,
                                       return_results=True, solver_tee=False)

            for k in loads.keys(): # revert back to initial loadings
                loads[k]['p_load'] = init_p_loads[k]
                loads[k]['q_load'] = init_q_loads[k]

            final_mult = mult
            print('mult={} has an acceptable solution.'.format(mult))
            break

        except Exception:
            print('mult={} raises an error. Continuing search.'.format(mult))

    if final_mult is None:
        print('Found no acceptable solutions with mult != 1. Try init between 1 and {}.'.format(mult))
        final_mult = 1

    return final_mult


def create_new_model_data(model_data, mult):
    md = model_data.clone_in_service()

    loads = dict(md.elements(element_type='load'))

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # multiply loads
    for k in loads.keys():
        loads[k]['p_load'] = init_p_loads[k] * mult
        loads[k]['q_load'] = init_q_loads[k] * mult

    md.data['system']['mult'] = mult

    return md


def inner_loop_solves(md_basepoint, md_flat, test_model_dict):
    '''
    solve models in test_model_dict (ideally, only one model is passed here)
    loads are multiplied by mult
    sensitivities from md_basepoint or md_flat as appropriate for the model being solved
    '''

    tm = test_model_dict

    if tm['acopf']:
        try:
            md_ac, m, results = solve_acopf(md_flat, "ipopt", return_model=True, return_results=True, solver_tee=False)
            record_results('acopf', md_ac)
        except Exception as e:
            print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            pass

    if tm['slopf']:
        try:
            md_lccm, m, results = solve_lccm(md_basepoint, "gurobi_persistent", return_model=True, return_results=True, solver_tee=False)
            record_results('slopf', md_lccm)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_default']:
        kwargs = {}
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, **kwargs)
            record_results('dlopf_default', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_lazy']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = True
        ptdf_options['lazy_voltage'] = True
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, options=options, **kwargs)
            record_results('dlopf_lazy', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_e5']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-5
        ptdf_options['abs_qtdf_tol'] = 5e-5
        ptdf_options['rel_vdf_tol'] = 10e-5
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, options=options, **kwargs)
            record_results('dlopf_e5', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_e4']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-4
        ptdf_options['abs_qtdf_tol'] = 5e-4
        ptdf_options['rel_vdf_tol'] = 10e-4
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, options=options, **kwargs)
            record_results('dlopf_e4', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_e3']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-3
        ptdf_options['abs_qtdf_tol'] = 5e-3
        ptdf_options['rel_vdf_tol'] = 10e-3
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, options=options, **kwargs)
            record_results('dlopf_e3', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dlopf_e2']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-2
        ptdf_options['abs_qtdf_tol'] = 5e-2
        ptdf_options['rel_vdf_tol'] = 10e-2
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdf, m, results = solve_fdf(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                           solver_tee=False, options=options, **kwargs)
            record_results('dlopf_e2', md_fdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_default']:
        kwargs = {}
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, **kwargs)
            record_results('clopf_default', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_lazy']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = True
        ptdf_options['lazy_voltage'] = True
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, options=options, **kwargs)
            record_results('clopf_lazy', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_e5']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-5
        ptdf_options['abs_qtdf_tol'] = 5e-5
        ptdf_options['rel_vdf_tol'] = 10e-5
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, options=options, **kwargs)
            record_results('clopf_e5', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_e4']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-4
        ptdf_options['abs_qtdf_tol'] = 5e-4
        ptdf_options['rel_vdf_tol'] = 10e-4
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, options=options, **kwargs)
            record_results('clopf_e4', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_e3']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-3
        ptdf_options['abs_qtdf_tol'] = 5e-3
        ptdf_options['rel_vdf_tol'] = 10e-3
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, options=options, **kwargs)
            record_results('clopf_e3', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_e2']:
        kwargs = {}
        options = {}
        #options['method'] = 1
        ptdf_options = {}
        ptdf_options['lazy'] = False
        ptdf_options['lazy_voltage'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-2
        ptdf_options['abs_qtdf_tol'] = 5e-2
        ptdf_options['rel_vdf_tol'] = 10e-2
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_fdfs, m, results = solve_fdf_simplified(md_basepoint, "gurobi_persistent", return_model=True, return_results=True,
                                                       solver_tee=False, options=options, **kwargs)
            record_results('clopf_e2', md_fdfs)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_default']:
        kwargs = {}
        ptdf_options = {}
        ptdf_options['lazy'] = False
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False, **kwargs)
            record_results('clopf_p_default', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_lazy']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = True
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False,
                                                      options=options, **kwargs)
            record_results('clopf_p_lazy', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_e5']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-5
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False,
                                                      options=options, **kwargs)
            record_results('clopf_p_e5', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_e4']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-4
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False,
                                                      options=options, **kwargs)
            record_results('clopf_p_e4', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_e3']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-3
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False,
                                                      options=options, **kwargs)
            record_results('clopf_p_e3', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['clopf_p_e2']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-2
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdfl, m, results = solve_dcopf_losses(md_basepoint, "gurobi_persistent",
                                                      dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                      return_model=True, return_results=True, solver_tee=False,
                                                      options=options, **kwargs)
            record_results('clopf_p_e2', md_ptdfl)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_default']:
        kwargs = {}
        ptdf_options = {}
        ptdf_options['lazy'] = False
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False, **kwargs)
            record_results('dcopf_ptdf_default', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_lazy']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = True
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False,
                                              options=options, **kwargs)
            record_results('dcopf_ptdf_lazy', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_e5']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-5
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False,
                                              options=options, **kwargs)
            record_results('dcopf_ptdf_e5', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_e4']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-4
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False,
                                              options=options, **kwargs)
            record_results('dcopf_ptdf_e4', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_e3']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-3
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False,
                                              options=options, **kwargs)
            record_results('dcopf_ptdf_e3', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_ptdf_e2']:
        kwargs = {}
        ptdf_options = {}
        options = {}
        #options['method'] = 1
        ptdf_options['lazy'] = False
        ptdf_options['abs_ptdf_tol'] = 1e-2
        kwargs['ptdf_options'] = ptdf_options
        try:
            md_ptdf, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_ptdf_dcopf_model,
                                              return_model=True, return_results=True, solver_tee=False,
                                              options=options, **kwargs)
            record_results('dcopf_ptdf_e2', md_ptdf)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['qcopf_btheta']:
        try:
            md_bthetal, m, results = solve_dcopf_losses(md_flat, "gurobi_persistent",
                                                        dcopf_losses_model_generator=create_btheta_losses_dcopf_model,
                                                        return_model=True, return_results=True, solver_tee=False)
            record_results('qcopf_btheta', md_bthetal)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e

    if tm['dcopf_btheta']:
        try:
            md_btheta, m, results = solve_dcopf(md_flat, "gurobi_persistent", dcopf_model_generator=create_btheta_dcopf_model,
                                                return_model=True, return_results=True, solver_tee=False)
            record_results('dcopf_btheta', md_btheta)
        except Exception as e:
            if 'infeasible' in str(e):
                print('...EXCEPTION OCCURRED: {}'.format(str(e)))
            else:
                raise e


def record_results(idx, md):
    '''
    writes model data (md) object to .json file
    '''

    data_utils_deprecated.destroy_dicts_of_fdf(md)

    mult = md.data['system']['mult']
    filename = md.data['system']['model_name'] + '_' + idx + '_{0:04.0f}'.format(mult * 1000)
    md.data['system']['filename'] = filename

    md.write_to_json(filename)
    print('...out: {}'.format(filename))


def get_solution_file_location(test_case):
    _, case = os.path.split(test_case)
    case, _ = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    solution_location = os.path.join(current_dir, 'transmission_test_instances', 'approximation_solution_files', case)

    return solution_location

def get_summary_file_location(folder):
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    location = os.path.join(current_dir, 'transmission_test_instances','approximation_summary_files', folder)

    if not os.path.exists(location):
        os.makedirs(location)

    return location


def create_testcase_directory(test_case):
    # directory locations
    cwd = os.getcwd()
    case_folder, case = os.path.split(test_case)
    case, ext = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))

    # move to case directory
    source = os.path.join(cwd, case + '_*.json')
    destination = get_solution_file_location(test_case)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        print('dest: {}'.format(destination))

        for src in glob.glob(source):
            print('src:  {}'.format(src))
            folder, file = os.path.split(src)
            dest = os.path.join(destination, file)  # full destination path will overwrite existing files
            shutil.move(src, dest)

    return destination


def solve_approximation_models(test_case, test_model_dict, init_min=0.9, init_max=1.1, steps=20):
    '''
    1. initialize base case and demand range
    2. loop over demand values
    3. record results to .json files
    '''

    _md_flat = create_ModelData(test_case)

    _md_basept, min_mult, max_mult = set_acopf_basepoint_min_max(_md_flat, init_min, init_max)
    test_model_dict['acopf'] = True

    ## put the sensitivities into modeData so they don't need to be recalculated for each model
    data_utils_deprecated.create_dicts_of_fdf_simplified(_md_basept)
    data_utils_deprecated.create_dicts_of_ptdf(_md_flat)

    # Calculate sensitivity multiplers, and make sure the base case mult=1 is included
    inc = (max_mult - min_mult) / steps
    multipliers = [round(min_mult + step * inc, 4) for step in range(0, steps + 1)]
    if 1.0 not in multipliers:
        multipliers.append(1.0)
        multipliers.sort()

    for mult in multipliers:
        md_basept = create_new_model_data(_md_basept, mult)
        md_flat = create_new_model_data(_md_flat, mult)
        inner_loop_solves(md_basept, md_flat, test_model_dict)

    create_testcase_directory(test_case)



def main(arg):

    #idxA0 = case_names.index('pglib_opf_case240_pserc')  ## < 1000 buses
    idxA = case_names.index('pglib_opf_case1354_pegase')  ## < 1000 buses
    idxB = case_names.index('pglib_opf_case2383wp_k')  ## 1354 - 2316 buses
    idxB = case_names.index('pglib_opf_case2736sp_k')  ## 1354 - 2316 buses
    idxC = case_names.index('pglib_opf_case6468_rte')  ## 2383 - 4661 buses
    idxD = case_names.index('pglib_opf_case13659_pegase')  ## 6468 - 10000 buses
    idxE = case_names.index('pglib_opf_case13659_pegase') + 1  ## 13659 buses

    if arg == 'A':
        idx_list = list(range(0,idxA))
    elif arg == 'B':
        idx_list = list(range(idxA,idxB))
    elif arg == 'C':
        idx_list = list(range(idxB,idxC))
    elif arg == 'D':
        idx_list = list(range(idxC,idxD))
    elif arg == 'E':
        idx_list = list(range(idxD,idxE))

    for idx in idx_list:
        submain(idx, show_plot=False, log_level=logging.INFO)


def submain(idx=None, show_plot=True, log_level=logging.ERROR):
    """
    solves models and generates plots for test case at test_cases[idx] or a default case
    """

    logger = logging.getLogger('egret')
    logger.setLevel(log_level)

    # Select default case
    if idx is None:
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case3_lmbd.m')
        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case5_pjm.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case30_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case24_ieee_rts.m')
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case118_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case300_ieee.m')
    else:
        test_case=idx_to_test_case(idx)

    # Select models to run
    test_model_dict = \
        {'acopf' : True,
         'slopf': True,
         'dlopf_default': True,
         'dlopf_lazy' : True,
         'dlopf_e5': True,
         'dlopf_e4': True,
         'dlopf_e3': True,
         'dlopf_e2': False,
         'clopf_default': True,
         'clopf_lazy': True,
         'clopf_e5': True,
         'clopf_e4': True,
         'clopf_e3': True,
         'clopf_e2': False,
         'clopf_p_default': True,
         'clopf_p_lazy': True,
         'clopf_p_e5': True,
         'clopf_p_e4': True,
         'clopf_p_e3': True,
         'clopf_p_e2': False,
         'qcopf_btheta': True,
         'dcopf_ptdf_default': True,
         'dcopf_ptdf_lazy': True,
         'dcopf_ptdf_e5': True,
         'dcopf_ptdf_e4': True,
         'dcopf_ptdf_e3': True,
         'dcopf_ptdf_e2': False,
         'dcopf_btheta': True
         }

    ## Model solves
    solve_approximation_models(test_case, test_model_dict, init_min=0.97, init_max=1.03, steps=6)

    ## Generate summary data
    spu.create_full_summary(test_case, test_model_dict)


def idx_to_test_case(s):
    try:
        idx = int(s)
        tc = test_cases[idx]
        return tc
    except IndexError:
        raise SyntaxError("Index out of range of test_cases.")
    except ValueError:
        try:
            idx = case_names.index(s)
            tc = test_cases[idx]
            return tc
        except ValueError:
            raise SyntaxError(
                "Expecting argument of either A, B, C, D, E, or an index or case name from the test_cases list.")

if __name__ == '__main__':
    import sys

    if len(sys.argv[1:]) == 1:
        if sys.argv[1] == "A" or \
                sys.argv[1] == "B" or \
                sys.argv[1] == "C" or \
                sys.argv[1] == "D" or \
                sys.argv[1] == "E":
            main(sys.argv[1])
        else:
            submain(sys.argv[1])
    else:
        submain()

#    raise SyntaxError("Expecting a single string argument: test_cases0, test_cases1, test_cases2, test_cases3, or test_cases4")
