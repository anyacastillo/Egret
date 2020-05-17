#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains several helper functions and classes that are useful when
modifying the data dictionary
"""
import os, shutil, glob, json
import egret.model_library.transmission.tx_utils as tx_utils
import egret.models.tests.ta_utils as tau
import egret.model_library.decl as decl
import pyomo.environ as pe
import pandas as pd
import numpy as np
from egret.models.acopf import create_psv_acopf_model
from egret.models.acpf import create_psv_acpf_model, solve_acpf
from egret.common.solver_interface import _solve_model
from pyomo.environ import value
import egret.data.data_utils_deprecated as data_utils_deprecated
from math import sqrt


def termination_condition(md):

    val = md.data['results']['termination']

    return val

def optimal(md):

    tc = md.data['results']['termination']
    if tc == 'optimal':
        return 1
    return 0

def infeasible(md):

    tc = md.data['results']['termination']
    if tc == 'infeasible':
        return 1
    return 0

def maxTimeLimit(md):

    tc = md.data['results']['termination']
    if tc == 'maxTimeLimit':
        return 1
    return 0

def maxIterations(md):

    tc = md.data['results']['termination']
    if tc == 'maxIterations':
        return 1
    return 0

def solverFailure(md):

    tc = md.data['results']['termination']
    if tc == 'solverFailure':
        return 1
    return 0

def internalSolverError(md):

    tc = md.data['results']['termination']
    if tc == 'internalSolverError':
        return 1
    return 0

def duals(md):
    try:
        val = md.data['results']['duals']
        return val
    except KeyError as e:
        print('...ModelData is missing: {}'.format(str(e)))
        return 0

def solve_time(md):

    if not optimal(md):
        return None
    val = md.data['results']['time']
    return val

def num_buses(md):

    bus_attrs = md.attributes(element_type='bus')
    val = len(bus_attrs['names'])

    return val


def num_branches(md):

    branch_attrs = md.attributes(element_type='branch')
    val = len(branch_attrs['names'])

    return val


def num_constraints(md):

    if not optimal(md):
        return None
    val = md.data['results']['#_cons']
    return val

def num_variables(md):

    if not optimal(md):
        return None
    val = md.data['results']['#_vars']
    return val

def num_nonzeros(md):

    if not optimal(md):
        return None
    results = md.data['results']

    if '#_nz' in results:
        val = results['#_nz']
        return val

    return None

def model_density(md):

    if not optimal(md):
        return None
    results = md.data['results']

    if '#_nz' in results:
        nc = results['#_cons']
        nv = results['#_vars']
        nz = results['#_nz']
        val = nz / ( nc * nv )
        return val

    return None


def total_cost(md):

    if not optimal(md):
        return None
    val = md.data['system']['total_cost']

    return val

def ploss(md):

    if not optimal(md):
        return None
    val = md.data['system']['ploss']

    return val

def qloss(md):

    if not optimal(md):
        return None
    val = md.data['system']['qloss']

    return val

def pgen(md):

    if not optimal(md):
        return None
    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['pg']

    return dispatch

def qgen(md):

    if not optimal(md):
        return None
    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['qg']

    return dispatch

def pflow(md):

    if not optimal(md):
        return None
    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['pf']

    return flow

def qflow(md):

    if not optimal(md):
        return None
    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['qf']

    return flow

def vmag(md):

    if not optimal(md):
        return None
    buses = dict(md.elements(element_type='bus'))
    vm = {}

    for b,bus in buses.items():
        vm[b] = bus['vm']

    return vm


def solve_infeas_model(model_data):
    # TODO: Change outputs from (summarized) slack & violation to (detailed) slack, violation, and error in ModelData

    # initial reference bus dispatch
    lin_gens = dict(model_data.elements(element_type='generator'))
    lin_buses = dict(model_data.elements(element_type='bus'))
    lin_branches = dict(model_data.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(lin_buses, lin_gens)
    ref_bus = model_data.data['system']['reference_bus']
    slack_p_init = sum(lin_gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])

    is_acopf = 'acopf' in model_data.data['system']['filename']

    # solve ACPF or return empty results and print exception message
    try:
        md, m, results = solve_acpf(model_data, "ipopt", return_results=True, return_model=True, solver_tee=False)
        termination = results.solver.termination_condition.__str__()
    except Exception as e:
        message = str(e)
        print('...EXCEPTION OCCURRED: {}'.format(message))
        if 'infeasible' in message:
            termination = 'infeasible'
            slack_p = None
            vm_UB_viol_dict = {}
            vm_LB_viol_dict = {}
            thermal_viol_dict = {}
            pf_error = {}
            qf_error = {}
            return slack_p, vm_UB_viol_dict, vm_LB_viol_dict, thermal_viol_dict, pf_error, qf_error, termination
        else:
            raise e

    vm_UB_viol_dict = dict()
    vm_LB_viol_dict = dict()
    thermal_viol_dict = dict()

    AC_gens = dict(md.elements(element_type='generator'))
    AC_buses = dict(md.elements(element_type='bus'))
    AC_branches = dict(md.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(AC_buses, AC_gens)
    buses_with_gens = tx_utils.buses_with_gens(AC_gens)

    # calculate change in slackbus P dispatch
    ref_bus = md.data['system']['reference_bus']
    slack_p_acpf = sum(AC_gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])
    slack_p = slack_p_acpf - slack_p_init

    # calculate voltage infeasibilities
    for bus_name, bus_dict in AC_buses.items():
        if bus_name != ref_bus and bus_name not in buses_with_gens:
            vm = bus_dict['vm']
            if vm > bus_dict['v_max']:
                vm_UB_viol_dict[bus_name] = vm - bus_dict['v_max']
            elif vm < bus_dict['v_min']:
                vm_LB_viol_dict[bus_name] = bus_dict['v_min'] - vm

    # calculate thermal infeasibilities
    for branch_name, branch_dict in AC_branches.items():
        sf = sqrt(branch_dict["pf"]**2 + branch_dict["qf"]**2)
        st = sqrt(branch_dict["pt"]**2 + branch_dict["qt"]**2)
        if sf > st: # to avoid double counting
            if sf > branch_dict['rating_long_term']:
                thermal_viol_dict[branch_name] = sf - branch_dict['rating_long_term']
        elif st > branch_dict['rating_long_term']:
            thermal_viol_dict[branch_name] = st - branch_dict['rating_long_term']

    # calculate flow errors
    pf_error = {}
    qf_error = {}
    for k, branch in lin_branches.items():
        if is_acopf:
            pf_ac = AC_branches[k]['pf']
            qf_ac = AC_branches[k]['qf']
        else:
            pf_ac = (AC_branches[k]['pf'] - AC_branches[k]['pt']) / 2
            qf_ac = (AC_branches[k]['qf'] - AC_branches[k]['qt']) / 2
        pf_error[k] = branch['pf'] - pf_ac
        if branch['qf'] is not None:
            qf_error[k] = branch['qf'] - qf_ac
        else:
            qf_error[k] = None

    return slack_p, vm_UB_viol_dict, vm_LB_viol_dict, thermal_viol_dict, pf_error, qf_error, termination

def get_infeas_from_model_data(md, infeas_name='acpf_slack', overwrite_existing=False):

    system_data = md.data['system']

    # repopulate data if not in the ModelData or an ovewrite is desired
    if infeas_name not in system_data.keys() or overwrite_existing:
        repopulate_acpf_to_modeldata(md)

    return system_data[infeas_name]

def repopulate_acpf_to_modeldata(md, abs_tol_vm=1e-6, rel_tol_therm=0.01):

    acpf_p_slack, vm_UB_viol, vm_LB_viol, thermal_viol, pf_error, qf_error, termination = solve_infeas_model(md)

    system_data = md.data['system']
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    num_bus = len(bus_attrs['names'])
    num_branch = len(branch_attrs['names'])

    s_max = branch_attrs['rating_long_term']
    thermal_list = []
    thermal_max = []
    for k in thermal_viol.keys():
        thermal_list.append(thermal_viol[k])
        thermal_max.append(s_max[k])

    vm_UB_list = list(vm_UB_viol.values())
    vm_LB_list = list(vm_LB_viol.values())
    vm_list = vm_UB_list + vm_LB_list

    pf_error_list = [p if p is not None else 0 for p in pf_error.values()]
    qf_error_list = [q if q is not None else 0 for q in qf_error.values()]

    ## save violations in ModelData
    system_data['acpf_termination'] = termination
    for k,viol in thermal_viol.items():
        branch = branches[k]
        branch['acpf_viol'] = viol
    for b,viol in vm_UB_viol.items():
        bus = buses[b]
        bus['acpf_viol'] = viol
    for b,viol in vm_LB_viol.items():
        bus = buses[b]
        bus['acpf_viol'] = -viol
    for k,branch in branches.items():
        branch['pf_error'] = pf_error[k]
        branch['qf_error'] = qf_error[k]

    ## save scalar data in ModelData
    system_data['acpf_slack'] = acpf_p_slack

    system_data['sum_vm_UB_viol'] = sum(vm_UB_list)
    system_data['sum_vm_LB_viol'] = sum(vm_LB_list)
    system_data['sum_vm_viol'] = sum(vm_list)
    system_data['sum_thermal_viol'] = sum(thermal_list)

    system_data['pf_error_1_norm'] = np.linalg.norm(pf_error_list,ord=1)
    system_data['qf_error_1_norm'] = np.linalg.norm(qf_error_list,ord=1)
    system_data['pf_error_inf_norm'] = np.linalg.norm(pf_error_list,ord=np.inf)
    system_data['qf_error_inf_norm'] = np.linalg.norm(pf_error_list,ord=np.inf)

    if len(vm_UB_list) > 0:
        system_data['avg_vm_UB_viol'] = sum(vm_UB_list) / len(vm_UB_list)
        system_data['max_vm_UB_viol'] = max(vm_UB_list)
    else:
        system_data['avg_vm_UB_viol'] = 0
        system_data['max_vm_UB_viol'] = 0

    if len(vm_LB_list) > 0:
        system_data['avg_vm_LB_viol'] = sum(vm_LB_list) / len(vm_LB_list)
        system_data['max_vm_LB_viol'] = max(vm_LB_list)
    else:
        system_data['avg_vm_LB_viol'] = 0
        system_data['max_vm_LB_viol'] = 0

    if len(vm_list) > 0:
        system_data['avg_vm_viol'] = sum(vm_list) / len(vm_list)
        system_data['max_vm_viol'] = max(vm_list)
    else:
        system_data['avg_vm_viol'] = 0
        system_data['max_vm_viol'] = 0

    if len(thermal_list) > 0:
        system_data['avg_thermal_viol'] = sum(thermal_list) / len(thermal_list)
        system_data['max_thermal_viol'] = max(thermal_list)
    else:
        system_data['avg_thermal_viol'] = 0
        system_data['max_thermal_viol'] = 0

    system_data['pct_vm_UB_viol'] = len([i for i in vm_UB_list if i > abs_tol_vm]) / num_bus
    system_data['pct_vm_LB_viol'] = len([i for i in vm_LB_list if i > abs_tol_vm]) / num_bus
    system_data['pct_vm_viol'] = len([i for i in vm_list if i > abs_tol_vm]) / num_bus
    system_data['pct_thermal_viol'] = len([t for i,t in enumerate(thermal_list)
                                           if t > rel_tol_therm * thermal_max[i]]) / num_branch

    if 'filename' in system_data.keys():
        data_utils_deprecated.destroy_dicts_of_fdf(md)
        filename = system_data['filename']
        model_name = system_data['model_name']
        md.write_to_json(filename)
        save_to_solution_directory(filename, model_name)
    else:
        print(system_data.keys())
        print('Failed to write modelData to json.')


def save_to_solution_directory(filename, model_name):

    # directory locations
    cwd = os.getcwd()
    source = os.path.join(cwd, filename + '.json')
    destination = tau.get_solution_file_location(model_name)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        #print('saving to dest: {}'.format(destination))

        for src in glob.glob(source):
            #print('src:  {}'.format(src))
            folder, file = os.path.split(src)
            dest = os.path.join(destination, file) # full destination path will overwrite existing files
            shutil.move(src, dest)

    return destination


def vm_UB_viol_sum(md):
    '''
    Returns the sum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_UB_viol')

    return sum_vm_UB_viol


def vm_LB_viol_sum(md):
    '''
    Returns the sum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_LB_viol')

    return sum_vm_LB_viol


def vm_viol_sum(md):
    '''
    Returns the sum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_viol')

    return sum_vm_viol


def thermal_viol_sum(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    sum_thermal_viol = get_infeas_from_model_data(md, infeas_name='sum_thermal_viol')

    return sum_thermal_viol


def vm_UB_viol_avg(md):
    '''
    Returns the average of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_UB_viol')

    return avg_vm_UB_viol

def vm_LB_viol_avg(md):
    '''
    Returns the average of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_LB_viol')

    return avg_vm_LB_viol


def vm_viol_avg(md):
    '''
    Returns the average of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_viol')

    return avg_vm_viol


def thermal_viol_avg(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    avg_thermal_viol = get_infeas_from_model_data(md, infeas_name='avg_thermal_viol')

    return avg_thermal_viol


def vm_UB_viol_max(md):
    '''
    Returns the maximum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='max_vm_UB_viol')

    return max_vm_UB_viol


def vm_LB_viol_max(md):
    '''
    Returns the maximum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='max_vm_LB_viol')

    return max_vm_LB_viol


def vm_viol_max(md):
    '''
    Returns the maximum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_viol = get_infeas_from_model_data(md, infeas_name='max_vm_viol')

    return max_vm_viol


def thermal_viol_max(md):
    '''
    Returns the maximum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    max_thermal_viol = get_infeas_from_model_data(md, infeas_name='max_thermal_viol')

    return max_thermal_viol


def vm_UB_viol_pct(md):
    '''
    Returns the number of voltage upper bound infeasibilites
    '''

    if not optimal(md):
        return None
    pct_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_UB_viol')

    return pct_vm_UB_viol


def vm_LB_viol_pct(md):
    '''
    Returns the number of voltage lower bound infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_LB_viol')

    return pct_vm_LB_viol


def vm_viol_pct(md):
    '''
    Returns the number of all voltage infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_viol')

    return pct_vm_viol


def thermal_viol_pct(md):
    '''
    Returns the number of thermal limit infeasibilites
    '''

    if not optimal(md):
        return None
    pct_thermal_viol = get_infeas_from_model_data(md, infeas_name='pct_thermal_viol')

    return pct_thermal_viol


def pf_error_1_norm(md):
    '''
    Returns the 1-norm of real power flow error
    '''

    if not optimal(md):
        return None
    pf_error_1_norm = get_infeas_from_model_data(md, infeas_name='pf_error_1_norm')

    return pf_error_1_norm


def qf_error_1_norm(md):
    '''
    Returns the 1-norm of reactive power flow error
    '''

    if not optimal(md):
        return None
    qf_error_1_norm = get_infeas_from_model_data(md, infeas_name='qf_error_1_norm')

    return qf_error_1_norm


def pf_error_inf_norm(md):
    '''
    Returns the infinity-norm of real power flow error
    '''

    if not optimal(md):
        return None
    pf_error_inf_norm = get_infeas_from_model_data(md, infeas_name='pf_error_inf_norm')

    return pf_error_inf_norm


def qf_error_inf_norm(md):
    '''
    Returns the infinity-norm of reactive power flow error
    '''

    if not optimal(md):
        return None
    qf_error_inf_norm = get_infeas_from_model_data(md, infeas_name='qf_error_inf_norm')

    return qf_error_inf_norm


def thermal_and_vm_viol_pct(md):

    p1 = thermal_viol_pct(md)
    p2 = vm_viol_pct(md)

    if p1 is None or p2 is not None:
        return None
    val = p1+p2

    return val

def acpf_slack(md):
    '''
    Returns the change in the slack bus real power dispatch in the ACPF solution in MW
    '''

    if not optimal(md):
        return None
    acpf_slack = get_infeas_from_model_data(md, infeas_name='acpf_slack')

    return acpf_slack

def thermal_viol(md):

    system_data = md.data['system']
    branches = dict(md.elements(element_type='branch'))

    if not optimal(md):
        return None

    if 'acpf_termination' in system_data.keys():
        if not system_data['acpf_termination'] == 'optimal':
            nan_dict = {b: np.nan for b in branches.keys()}
            return nan_dict
        else:
            pass
    else:
        print('Repopulating ACPF violations for {}.'.format(system_data['filename']))
        repopulate_acpf_to_modeldata(md)

    viol = {}

    for b,branch in branches.items():
        if 'acpf_viol' in branch.keys():
            viol[b] = branch['acpf_viol']
        else:
            viol[b] = 0

    return viol

def vm_viol(md):

    system_data = md.data['system']
    buses = dict(md.elements(element_type='bus'))

    if not optimal(md):
        return None

    if 'acpf_termination' in system_data.keys():
        if not system_data['acpf_termination'] == 'optimal':
            nan_dict = {b: np.nan for b in buses.keys()}
            return nan_dict
        else:
            pass
    else:
        print('Repopulating ACPF violations for {}.'.format(system_data['filename']))
        repopulate_acpf_to_modeldata(md)

    viol = {}

    for b,bus in buses.items():
        if 'acpf_viol' in bus.keys():
            viol[b] = bus['acpf_viol']
        else:
            viol[b] = 0

    return viol

def acpf_error(md,key='pf_error'):

    system_data = md.data['system']
    branches = dict(md.elements(element_type='branch'))
    bn = [k for k in branches.keys()]
    branch_keys = [k for k in branches[bn[1]].keys()]

    if not optimal(md):
        return None

    if 'acpf_termination' in system_data.keys() and key in branch_keys:
        if not system_data['acpf_termination'] == 'optimal':
            nan_dict = {b: np.nan for b in branches.keys()}
            return nan_dict
        else:
            pass
    else:
        print('Repopulating ACPF violations for {}.'.format(system_data['filename']))
        repopulate_acpf_to_modeldata(md)

    error = {}

    for k,branch in branches.items():
        if key in branch.keys():
            error[k] = branch[key]
        else:
            error[k] = 0

    return error

def pf_error(md):
    error = acpf_error(md, key='pf_error')
    return error

def qf_error(md):
    error = acpf_error(md, key='qf_error')
    return error
