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

import os, shutil, glob, json, gc
import sys, getopt
import pandas as pd
import math
import unittest
import logging
import copy
import egret.data.test_utils as tu
import egret.data.summary_plot_utils as spu
from pyomo.opt import SolverFactory, TerminationCondition
from egret.common.log import logger
from egret.models.acopf import solve_acopf, create_psv_acopf_model
from egret.models.ccm import solve_ccm, create_ccm_model
from egret.models.fdf import solve_fdf, create_fdf_model
from egret.models.fdf_simplified import solve_fdf_simplified, create_simplified_fdf_model
from egret.models.lccm import solve_lccm, create_lccm_model
from egret.models.dcopf_losses import solve_dcopf_losses, create_btheta_losses_dcopf_model, create_ptdf_losses_dcopf_model
from egret.models.dcopf import solve_dcopf, create_btheta_dcopf_model, create_ptdf_dcopf_model
from egret.models.copperplate_dispatch import solve_copperplate_dispatch, create_copperplate_dispatch_approx_model
from egret.models.tests.ta_utils import *
from egret.data.data_utils_deprecated import destroy_dicts_of_fdf, create_dicts_of_ptdf, create_dicts_of_fdf
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join

current_dir = os.path.dirname(os.path.abspath(__file__))
# test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]
#test_cases = [os.path.join(current_dir, 'download', 'pglib-opf-master', '{}.m'.format(i)) for i in case_names]


test_model_list = [
    'acopf',
    'slopf',
    'dlopf_full',
    'dlopf_e4',
    'dlopf_e2',
    'dlopf_lazy_full',
    'dlopf_lazy_e4',
    'dlopf_lazy_e2',
    'clopf_full',
    'clopf_e4',
    'clopf_e2',
    'clopf_lazy_full',
    'clopf_lazy_e4',
    'clopf_lazy_e2',
    'plopf_full',
    'plopf_e4',
    'plopf_e2',
    'plopf_lazy_full',
    'plopf_lazy_e4',
    'plopf_lazy_e2',
    'ptdf_full',
    'ptdf_e4',
    'ptdf_e2',
    'ptdf_lazy_full',
    'ptdf_lazy_e4',
    'ptdf_lazy_e2',
    'btheta',
    # 'btheta_qcp',
]


def generate_test_model_dict(test_model_list):

    test_model_dict = {}
    _kwargs = {'return_model' :False, 'return_results' : False, 'solver_tee' : False}
    tol_keys = ['rel_ptdf_tol', 'rel_qtdf_tol', 'rel_pldf_tol', 'rel_qldf_tol', 'rel_vdf_tol']


    for tm in test_model_list:
        # create empty settings dictionary for each model type
        tmd = dict()
        tmd['kwargs'] = copy.deepcopy(_kwargs)

        # Build ptdf_options based on model name
        _ptdf_options = dict()
        if 'lazy' in tm:
            _ptdf_options['lazy'] = True
            if 'dlopf' in tm or 'clopf' in tm:
                _ptdf_options['lazy_voltage'] = True
        tol = None
        if 'e5' in tm:
            tol = 1e-5
        elif 'e4' in tm:
            tol = 1e-4
        elif 'e3' in tm:
            tol = 1e-3
        elif 'e2' in tm:
            tol = 1e-2
        if any(e in tm for e in ['e5','e4','e3','e2']):
            for k in tol_keys:
                _ptdf_options[k] = tol

        if 'acopf' in tm:
            tmd['solve_func'] = solve_acopf
            tmd['initial_solution'] = 'flat'
            tmd['solver'] = 'ipopt'

        elif 'slopf' in tm:
            tmd['solve_func'] = solve_lccm
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            #tmd['kwargs']['include_pf_feasibility_slack'] = True
            #tmd['kwargs']['include_qf_feasibility_slack'] = True
            tmd['kwargs']['include_v_feasibility_slack'] = True

        elif 'dlopf' in tm:
            tmd['solve_func'] = solve_fdf
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            #tmd['kwargs']['include_pf_feasibility_slack'] = True
            #tmd['kwargs']['include_qf_feasibility_slack'] = True
            tmd['kwargs']['include_v_feasibility_slack'] = True

        elif 'clopf' in tm:
            tmd['solve_func'] = solve_fdf_simplified
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            #tmd['kwargs']['include_pf_feasibility_slack'] = True
            #tmd['kwargs']['include_qf_feasibility_slack'] = True
            tmd['kwargs']['include_v_feasibility_slack'] = True

        elif 'plopf' in tm:
            tmd['solve_func'] = solve_dcopf_losses
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['dcopf_losses_model_generator'] = create_ptdf_losses_dcopf_model
            #tmd['kwargs']['include_pf_feasibility_slack'] = True

        elif 'ptdf' in tm:
            tmd['solve_func'] = solve_dcopf
            tmd['initial_solution'] = 'flat'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['dcopf_model_generator'] = create_ptdf_dcopf_model
            #tmd['kwargs']['include_pf_feasibility_slack'] = True

        elif 'btheta' in tm:
            if 'qcp' in tm:
                tmd['solve_func'] = solve_dcopf_losses
                tmd['intial_solution'] = 'flat'
                tmd['solver'] = 'gurobi_persistent'
                tmd['kwargs']['dcopf_losses_model_generator'] = create_btheta_losses_dcopf_model
            else:
                tmd['solve_func'] = solve_dcopf
                tmd['initial_solution'] = 'flat'
                tmd['solver'] = 'gurobi_persistent'
                tmd['kwargs']['dcopf_model_generator'] = create_btheta_dcopf_model

            #tmd['kwargs']['include_pf_feasibility_slack'] = True

        # settings to suppress non-lazy D-LOPF and C-LOPF models in large (>1,000 bus) cases
        dense_models = ['dlopf', 'clopf', 'plopf', 'ptdf']
        if any(dm in tm for dm in dense_models) and 'lazy' not in tm:
            #tmd['suppress_large_cases'] = True
            tmd['suppress_large_cases'] = False     # switch comment with previous line to suppress DF model solves of large cases
        else:
            tmd['suppress_large_cases'] = False

        test_model_dict[tm] = copy.deepcopy(tmd)

    return test_model_dict



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


def inner_loop_solves(md_basepoint, md_flat, test_model_list):
    '''
    solve models in test_model_dict (ideally, only one model is passed here)
    loads are multiplied by mult
    sensitivities from md_basepoint or md_flat as appropriate for the model being solved
    '''

    test_model_dict = generate_test_model_dict(test_model_list)
    bus_attrs = md_flat.attributes(element_type='bus')
    num_bus = len(bus_attrs['names'])

    for tm in test_model_list:

        tm_dict = test_model_dict[tm]

        if tm_dict['suppress_large_cases'] and num_bus > 1000:
            continue

        print('>>>>> BEGIN SOLVE: {} <<<<<'.format(tm))

        solve_func = tm_dict['solve_func']
        initial_solution = tm_dict['initial_solution']
        solver = tm_dict['solver']
        kwargs = tm_dict['kwargs']

        if initial_solution == 'flat':
            md_input = md_flat
        elif initial_solution == 'basepoint':
            md_input = md_basepoint
        else:
            raise Exception('test_model_dict must provide valid inital_solution')

        try:
            md_out = solve_func(md_input, solver=solver, **kwargs)
            logger.critical('\t COST = ${:,.2f}'.format(md_out.data['system']['total_cost']))
            logger.critical('\t TIME = {:.5f} seconds'.format(md_out.data['results']['time']))
        except Exception as e:
            md_out = md_input
            md_out.data['results'] = {}
            message = str(e)
            print('...EXCEPTION OCCURRED: {}'.format(message))
            if 'infeasible' in message:
                md_out.data['results']['termination'] = 'infeasible'
            else:
                md_out.data['results']['termination'] = 'other'

        record_results(tm, md_out)


def record_results(idx, md):
    '''
    writes model data (md) object to .json file
    '''

    destroy_dicts_of_fdf(md)

    mult = md.data['system']['mult']
    filename = md.data['system']['model_name'] + '_' + idx + '_{0:04.0f}'.format(mult * 1000)
    md.data['system']['filename'] = filename

    md.write_to_json(filename)
    print('...out: {}.json'.format(filename))

    if md.data['results']['termination'] == 'optimal':
        del md
        gc.collect()
    else:
        del md.data['results']
        del md.data['system']['filename']
        gc.collect()


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


def solve_approximation_models(test_case, test_model_list, init_min=0.9, init_max=1.1, steps=20):
    '''
    1. initialize base case and demand range
    2. loop over demand values
    3. record results to .json files
    '''

    _md_flat = create_ModelData(test_case)

    _md_basept, min_mult, max_mult = set_acopf_basepoint_min_max(_md_flat, init_min, init_max)
    if 'acopf' not in test_model_list:
        test_model_list.append('acopf')

    ## put the sensitivities into modeData so they don't need to be recalculated for each model
    create_dicts_of_fdf(_md_basept)
    create_dicts_of_ptdf(_md_flat)

    # Calculate sensitivity multiplers, and make sure the base case mult=1 is included
    inc = (max_mult - min_mult) / steps
    multipliers = [round(min_mult + step * inc, 4) for step in range(0, steps + 1)]
    if 1.0 not in multipliers:
        multipliers.append(1.0)
        multipliers.sort()

    for mult in multipliers:
        md_basept = create_new_model_data(_md_basept, mult)
        md_flat = create_new_model_data(_md_flat, mult)
        inner_loop_solves(md_basept, md_flat, test_model_list)

    create_testcase_directory(test_case)



def run_test_loop(idx=None, show_plot=False, log_level=logging.CRITICAL):
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

    ## Model solves
    solve_approximation_models(test_case, test_model_list, init_min=0.9, init_max=1.1, steps=20)

    ## Generate summary data
    spu.create_full_summary(test_case, test_model_list, show_plot=show_plot)


def run_nominal_test(idx=None, show_plot=False, log_level=logging.CRITICAL):
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

    ## Model solves
    md_flat = create_ModelData(test_case)
    print('>>>>> BEGIN SOLVE: acopf <<<<<')
    md_basept = solve_acopf(md_flat,solver='ipopt',solver_tee=False)
    logger.critical('\t COST = ${:,.2f}'.format(md_basept.data['system']['total_cost']))
    logger.critical('\t TIME = {:.5f} seconds'.format(md_basept.data['results']['time']))
    md_basept.data['system']['mult'] = 1
    record_results('acopf', md_basept)

    if 'acopf' in test_model_list:
        test_model_list.remove('acopf')

    ## put the sensitivities into modeData so they don't need to be recalculated for each model
    create_dicts_of_fdf(md_basept)
    create_dicts_of_ptdf(md_flat)
    md_basept = create_new_model_data(md_basept, 1.0)
    md_flat = create_new_model_data(md_flat, 1.0)

    inner_loop_solves(md_basept, md_flat, test_model_list)

    create_testcase_directory(test_case)

    summarize_nominal_data(test_case=test_case, show_plot=show_plot)

def summarize_nominal_data(idx=0, test_case=None,show_plot=True, log_level=None):

    if test_case is None:
        test_case = idx_to_test_case(idx)

    spu.update_data_file(test_case)
    spu.update_data_tables()
    spu.acpf_violations_plot(test_case, test_model_list, show_plot=show_plot)


def batch(arg, subbatch=run_test_loop):

    idxA0 = 0
    #idxA0 = case_names.index('pglib_opf_case179_goc')  ## redefine first case of A
    idxA = case_names.index('pglib_opf_case1354_pegase')  ## < 1000 buses
    idxB = case_names.index('pglib_opf_case2736sp_k')  ## 1354 - 2383 buses
    idxC = case_names.index('pglib_opf_case6468_rte')  ## 2383 - 4661 buses
    idxD = case_names.index('pglib_opf_case13659_pegase')  ## 6468 - 10000 buses
    idxE = case_names.index('pglib_opf_case13659_pegase') + 1  ## 13659 buses

    if arg == 'A':
        idx_list = list(range(idxA0,idxA))
    elif arg == 'B':
        idx_list = list(range(idxA,idxB))
    elif arg == 'C':
        idx_list = list(range(idxB,idxC))
    elif arg == 'D':
        idx_list = list(range(idxC,idxD))
    elif arg == 'E':
        idx_list = list(range(idxD,idxE))

    for idx in idx_list:
        subbatch(idx, show_plot=False, log_level=logging.CRITICAL)


def main(argv):

    message = 'test_approximations.py usage must include a batch or a case: \n'
    message += '  -b --batch=<A,B,C,D,E>      ::  run a preset batch of cases \n'
    message += '  -c --case=<case_name/idx>   ::  run a specific case or case index \n'
    message += '  -d --data-only              ::  run summary data only \n'
    message += '  -n --nominal                ::  run a specific case with nominal demand only \n'
    message += '  -q --quick                  ::  same as nominal option \n'

    try:
        opts, args = getopt.getopt(argv, "b:c:dnq", ["batch=", "case=", "data", "nominal", "quick"])
    except getopt.GetoptError:
        print(message)
        sys.exit(2)

    batch_run = False
    quick_run = False
    data_run = False
    case_run = False
    for opt, arg in opts:
        if opt in ('-b', '--batch'):
            batch_run = True
            batch_arg = arg
        elif opt in ('-c', '--case'):
            case_run = True
            case_arg = arg
        elif opt in ('-d', '--data-only'):
            data_run = True
        elif opt in ('-n', '--nominal'):
            quick_run = True
        elif opt in ('-q', '--quick'):
            quick_run = True

    if batch_run:
        if quick_run:
            batch(batch_arg, subbatch=run_nominal_test)
        elif data_run:
            batch(batch_arg, subbatch=summarize_nominal_data)
        else:
            batch(batch_arg, subbatch=run_test_loop)
    elif case_run:
        if quick_run:
            run_nominal_test(case_arg)
        elif data_run:
            summarize_nominal_data(idx=case_arg)
        else:
            run_test_loop(case_arg)
    else:
        print(message)


if __name__ == '__main__':
    main(sys.argv[1:])
