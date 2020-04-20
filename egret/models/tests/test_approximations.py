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
from egret.models.tests.ta_utils import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join

current_dir = os.path.dirname(os.path.abspath(__file__))
# test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]
#test_cases = [os.path.join(current_dir, 'download', 'pglib-opf-master', '{}.m'.format(i)) for i in case_names]

_kwargs = {'return_model' :True, 'return_results' : True, 'solver_tee' : False}
e5_options = {'abs_ptdf_tol' : 1e-5, 'abs_qtdf_tol' : 5e-5, 'rel_vdf_tol' : 10e-5}
e4_options = {'abs_ptdf_tol' : 1e-4, 'abs_qtdf_tol' : 5e-4, 'rel_vdf_tol' : 10e-4}
e3_options = {'abs_ptdf_tol' : 1e-3, 'abs_qtdf_tol' : 5e-3, 'rel_vdf_tol' : 10e-3}
e2_options = {'abs_ptdf_tol' : 1e-2, 'abs_qtdf_tol' : 5e-2, 'rel_vdf_tol' : 10e-2}

test_model_dict = {}
tmd = test_model_dict

tmd['acopf'] = {}
tmd['acopf']['solve_func'] = solve_acopf
tmd['acopf']['intial_solution'] = 'flat'
tmd['acopf']['solver'] = 'ipopt'
tmd['acopf']['kwargs'] = _kwargs

tmd['slopf'] = {}
tmd['slopf']['solve_func'] = solve_lccm
tmd['slopf']['intial_solution'] = 'basepoint'
tmd['slopf']['solver'] = 'gurobi_persistent'
tmd['slopf']['kwargs'] = _kwargs

tmd['dlopf_default'] = {}
tmd['dlopf_default']['solve_func'] = solve_fdf
tmd['dlopf_default']['intial_solution'] = 'basepoint'
tmd['dlopf_default']['solver'] = 'gurobi_persistent'
tmd['dlopf_default']['kwargs'] = _kwargs

tmd['dlopf_lazy'] = {}
tmd['dlopf_lazy']['solve_func'] = solve_fdf
tmd['dlopf_lazy']['intial_solution'] = 'basepoint'
tmd['dlopf_lazy']['solver'] = 'gurobi_persistent'
tmd['dlopf_lazy']['kwargs'] = {**_kwargs, 'ptdf_options':{'lazy' : True, 'lazy_voltage' : True} }

tmd['dlopf_e5'] = {}
tmd['dlopf_e5']['solve_func'] = solve_fdf
tmd['dlopf_e5']['intial_solution'] = 'basepoint'
tmd['dlopf_e5']['solver'] = 'gurobi_persistent'
tmd['dlopf_e5']['kwargs'] = {**_kwargs, 'ptdf_options' : e5_options}

tmd['dlopf_e4'] = {}
tmd['dlopf_e4']['solve_func'] = solve_fdf
tmd['dlopf_e4']['intial_solution'] = 'basepoint'
tmd['dlopf_e4']['solver'] = 'gurobi_persistent'
tmd['dlopf_e4']['kwargs'] = {**_kwargs, 'ptdf_options' : e4_options}

tmd['dlopf_e3'] = {}
tmd['dlopf_e3']['solve_func'] = solve_fdf
tmd['dlopf_e3']['intial_solution'] = 'basepoint'
tmd['dlopf_e3']['solver'] = 'gurobi_persistent'
tmd['dlopf_e3']['kwargs'] = {**_kwargs, 'ptdf_options' : e3_options}

tmd['dlopf_e2'] = {}
tmd['dlopf_e2']['solve_func'] = solve_fdf
tmd['dlopf_e2']['intial_solution'] = 'basepoint'
tmd['dlopf_e2']['solver'] = 'gurobi_persistent'
tmd['dlopf_e2']['kwargs'] = {**_kwargs, 'ptdf_options' : e2_options}

tmd['clopf_default'] = {}
tmd['clopf_default']['solve_func'] = solve_fdf_simplified
tmd['clopf_default']['intial_solution'] = 'basepoint'
tmd['clopf_default']['solver'] = 'gurobi_persistent'
tmd['clopf_default']['kwargs'] = _kwargs

tmd['clopf_lazy'] = {}
tmd['clopf_lazy']['solve_func'] = solve_fdf_simplified
tmd['clopf_lazy']['intial_solution'] = 'basepoint'
tmd['clopf_lazy']['solver'] = 'gurobi_persistent'
tmd['clopf_lazy']['kwargs'] = {**_kwargs, 'ptdf_options':{'lazy' : True, 'lazy_voltage' : True} }

tmd['clopf_e5'] = {}
tmd['clopf_e5']['solve_func'] = solve_fdf_simplified
tmd['clopf_e5']['intial_solution'] = 'basepoint'
tmd['clopf_e5']['solver'] = 'gurobi_persistent'
tmd['clopf_e5']['kwargs'] = {**_kwargs, 'ptdf_options' : e5_options}

tmd['clopf_e4'] = {}
tmd['clopf_e4']['solve_func'] = solve_fdf_simplified
tmd['clopf_e4']['intial_solution'] = 'basepoint'
tmd['clopf_e4']['solver'] = 'gurobi_persistent'
tmd['clopf_e4']['kwargs'] = {**_kwargs, 'ptdf_options' : e4_options}

tmd['clopf_e3'] = {}
tmd['clopf_e3']['solve_func'] = solve_fdf_simplified
tmd['clopf_e3']['intial_solution'] = 'basepoint'
tmd['clopf_e3']['solver'] = 'gurobi_persistent'
tmd['clopf_e3']['kwargs'] = {**_kwargs, 'ptdf_options' : e3_options}

tmd['clopf_e2'] = {}
tmd['clopf_e2']['solve_func'] = solve_fdf_simplified
tmd['clopf_e2']['intial_solution'] = 'basepoint'
tmd['clopf_e2']['solver'] = 'gurobi_persistent'
tmd['clopf_e2']['kwargs'] = {**_kwargs, 'ptdf_options' : e2_options}

tmd['plopf_default'] = {}
tmd['plopf_default']['solve_func'] = solve_dcopf_losses
tmd['plopf_default']['intial_solution'] = 'basepoint'
tmd['plopf_default']['solver'] = 'gurobi_persistent'
tmd['plopf_default']['kwargs'] = {**_kwargs,
                                    'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model}

tmd['plopf_lazy'] = {}
tmd['plopf_lazy']['solve_func'] = solve_dcopf_losses
tmd['plopf_lazy']['intial_solution'] = 'basepoint'
tmd['plopf_lazy']['solver'] = 'gurobi_persistent'
tmd['plopf_lazy']['kwargs'] = {**_kwargs, 'ptdf_options':{'lazy' : True},
                               'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model}

tmd['plopf_e5'] = {}
tmd['plopf_e5']['solve_func'] = solve_dcopf_losses
tmd['plopf_e5']['intial_solution'] = 'basepoint'
tmd['plopf_e5']['solver'] = 'gurobi_persistent'
tmd['plopf_e5']['kwargs'] = {**_kwargs, 'ptdf_options' : e5_options,
                               'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model}

tmd['plopf_e4'] = {}
tmd['plopf_e4']['solve_func'] = solve_dcopf_losses
tmd['plopf_e4']['intial_solution'] = 'basepoint'
tmd['plopf_e4']['solver'] = 'gurobi_persistent'
tmd['plopf_e4']['kwargs'] = {**_kwargs, 'ptdf_options' : e4_options,
                               'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model}

tmd['plopf_e3'] = {}
tmd['plopf_e3']['solve_func'] = solve_dcopf_losses
tmd['plopf_e3']['intial_solution'] = 'basepoint'
tmd['plopf_e3']['solver'] = 'gurobi_persistent'
tmd['plopf_e3']['kwargs'] = {**_kwargs, 'ptdf_options' : e3_options,
                               'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model}

tmd['plopf_e2'] = {}
tmd['plopf_e2']['solve_func'] = solve_dcopf_losses
tmd['plopf_e2']['intial_solution'] = 'basepoint'
tmd['plopf_e2']['solver'] = 'gurobi_persistent'
tmd['plopf_e2']['kwargs'] = {**_kwargs, 'ptdf_options' : e2_options,
                               'dcopf_losses_model_generator' : create_ptdf_losses_dcopf_model }

tmd['qcopf_btheta'] = {}
tmd['qcopf_btheta']['solve_func'] = solve_dcopf_losses
tmd['qcopf_btheta']['intial_solution'] = 'flat'
tmd['qcopf_btheta']['solver'] = 'gurobi_persistent'
tmd['qcopf_btheta']['kwargs'] = {**_kwargs,
                               'dcopf_losses_model_generator' : create_btheta_losses_dcopf_model }

tmd['dcopf_ptdf_default'] = {}
tmd['dcopf_ptdf_default']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_default']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_default']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_default']['kwargs'] = {**_kwargs,
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_ptdf_lazy'] = {}
tmd['dcopf_ptdf_lazy']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_lazy']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_lazy']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_lazy']['kwargs'] = {**_kwargs, 'ptdf_options':{'lazy' : True},
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_ptdf_e5'] = {}
tmd['dcopf_ptdf_e5']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_e5']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_e5']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_e5']['kwargs'] = {**_kwargs, 'ptdf_options' : e5_options,
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_ptdf_e4'] = {}
tmd['dcopf_ptdf_e4']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_e4']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_e4']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_e4']['kwargs'] = {**_kwargs, 'ptdf_options' : e4_options,
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_ptdf_e3'] = {}
tmd['dcopf_ptdf_e3']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_e3']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_e3']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_e3']['kwargs'] = {**_kwargs, 'ptdf_options' : e3_options,
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_ptdf_e2'] = {}
tmd['dcopf_ptdf_e2']['solve_func'] = solve_dcopf
tmd['dcopf_ptdf_e2']['intial_solution'] = 'flat'
tmd['dcopf_ptdf_e2']['solver'] = 'gurobi_persistent'
tmd['dcopf_ptdf_e2']['kwargs'] = {**_kwargs, 'ptdf_options' : e2_options,
                               'dcopf_model_generator' : create_ptdf_dcopf_model }

tmd['dcopf_btheta'] = {}
tmd['dcopf_btheta']['solve_func'] = solve_dcopf
tmd['dcopf_btheta']['intial_solution'] = 'flat'
tmd['dcopf_btheta']['solver'] = 'gurobi_persistent'
tmd['dcopf_btheta']['kwargs'] = {**_kwargs,
                               'dcopf_model_generator' : create_btheta_dcopf_model }



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

    for tm in test_model_list:

        tm_dict = test_model_dict[tm]

        solve_func = tm_dict['solve_func']
        intial_solution = tm_dict['intial_solution']
        solver = tm_dict['solver']
        kwargs = tm_dict['kwargs']

        if intial_solution == 'flat':
            md_input = md_flat
        elif intial_solution == 'basepoint':
            md_input = md_basepoint
        else:
            raise Exception('test_model_dict must provide valid inital_solution')

        try:
            md_out,m,results = solve_func(md_input, solver, **kwargs)
        except Exception as e:
            md_out = md_input.clone()
            message = str(e)
            print('...EXCEPTION OCCURRED: {}'.format(message))
            if 'infeasible' in message:
                md_out.data['system']['termination'] = 'infeasible'
            else:
                md_out.data['system']['termination'] = 'other'

        record_results(tm, md_out)


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
        inner_loop_solves(md_basept, md_flat, test_model_list)

    create_testcase_directory(test_case)



def main(arg):

    idxA0 = 0
    #idxA0 = case_names.index('pglib_opf_case89_pegase')  ## redefine first case of A
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
    test_model_list = [
         'acopf',
         'slopf',
         'dlopf_default',
         'dlopf_lazy',
         'dlopf_e5',
         'dlopf_e4',
         'dlopf_e3',
         #'dlopf_e2',
         'clopf_default',
         'clopf_lazy',
         'clopf_e5',
         'clopf_e4',
         'clopf_e3',
         #'clopf_e2',
         'plopf_default',
         'plopf_lazy',
         'plopf_e5',
         'plopf_e4',
         'plopf_e3',
         #'plopf_e2',
         'qcopf_btheta',
         'dcopf_ptdf_default',
         'dcopf_ptdf_lazy',
         'dcopf_ptdf_e5',
         'dcopf_ptdf_e4',
         'dcopf_ptdf_e3',
         #'dcopf_ptdf_e2',
         'dcopf_btheta'
         ]

    ## Model solves
    #solve_approximation_models(test_case, test_model_list, init_min=0.9, init_max=1.1, steps=20)

    ## Generate summary data
    spu.create_full_summary(test_case, test_model_list, show_plot=show_plot)


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
