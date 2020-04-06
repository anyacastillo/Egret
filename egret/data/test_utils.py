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
    val = md.data['results']['duals']
    return val

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

    # initial reference bus dispatch
    gens = dict(model_data.elements(element_type='generator'))
    buses = dict(model_data.elements(element_type='bus'))
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    ref_bus = model_data.data['system']['reference_bus']
    slack_p_init = sum(gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])

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
            return slack_p, vm_UB_viol_dict, vm_LB_viol_dict, thermal_viol_dict, termination
        else:
            raise e

    vm_UB_viol_dict = dict()
    vm_LB_viol_dict = dict()
    thermal_viol_dict = dict()

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    buses_with_gens = tx_utils.buses_with_gens(gens)

    # calculate change in slackbus P dispatch
    ref_bus = md.data['system']['reference_bus']
    slack_p_acpf = sum(gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])
    slack_p = slack_p_acpf - slack_p_init

    # calculate voltage infeasibilities
    for bus_name, bus_dict in buses.items():
        if bus_name != ref_bus and bus_name not in buses_with_gens:
            vm = bus_dict['vm']
            if vm > bus_dict['v_max']:
                vm_UB_viol_dict[bus_name] = vm - bus_dict['v_max']
            elif vm < bus_dict['v_min']:
                vm_LB_viol_dict[bus_name] = bus_dict['v_min'] - vm

    # calculate thermal infeasibilities
    for branch_name, branch_dict in branches.items():
        sf = sqrt(branch_dict["pf"]**2 + branch_dict["qf"]**2)
        st = sqrt(branch_dict["pt"]**2 + branch_dict["qt"]**2)
        if sf > st: # to avoid double counting
            if sf > branch_dict['rating_long_term']:
                thermal_viol_dict[branch_name] = sf - branch_dict['rating_long_term']
        elif st > branch_dict['rating_long_term']:
            thermal_viol_dict[branch_name] = st - branch_dict['rating_long_term']

    return slack_p, vm_UB_viol_dict, vm_LB_viol_dict, thermal_viol_dict, termination

def get_infeas_from_model_data(md, infeas_name='acpf_slack', overwrite_existing=False, abs_tol_vm=1e-6, rel_tol_therm=0.01):

    system_data = md.data['system']

    # return data if it is already in the ModelData and an ovewrite is not desired
    if infeas_name in system_data.keys() and not overwrite_existing:
        return system_data[infeas_name]

    ## printing some status updates
    if 'filename' in system_data:
        name = system_data['filename']
    elif 'mult' in system_data:
        name = 'mult={}'.format(system_data['mult'])
    else:
        name = 'file'

    message = '...'
    if not infeas_name in system_data.keys():
        message += 'Did not find ' + infeas_name + ' in '+ name+ '. '
    else:
        show_me = pd.DataFrame(system_data,index=[name])
        #print('...existing system data: {}'.format(show_me.T))

    # otherwise, solve the sum_infeas model and save solution to md
    acpf_p_slack, vm_UB_viol, vm_LB_viol, thermal_viol, termination = solve_infeas_model(md)
    message += 'Infeasibility model returned ' + termination + '.'
    print(message)
    #m_ac.pprint()

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
#    thermal_list = list(thermal_viol.values())

    system_data['acpf_slack'] = acpf_p_slack

    system_data['sum_vm_UB_viol'] = sum(vm_UB_list)
    system_data['sum_vm_LB_viol'] = sum(vm_LB_list)
    system_data['sum_vm_viol'] = sum(vm_list)
    system_data['sum_thermal_viol'] = sum(thermal_list)

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

    show_me = pd.DataFrame(system_data,index=[name])
    #print('...overwriting system data: {}'.format(show_me.T))

    if 'filename' in system_data.keys():
        data_utils_deprecated.destroy_dicts_of_fdf(md)
        filename = system_data['filename']
        model_name = system_data['model_name']
        md.write_to_json(filename)
        save_to_solution_directory(filename,model_name)
    else:
        print(system_data.keys())
        print('Failed to write modelData to json.')

    if infeas_name in system_data.keys():
        return system_data[infeas_name]
    else:
        print(system_data.keys())
        raise NameError('Quantity '+ infeas_name +' not found in sum_infeas model data.')

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


def sum_vm_UB_viol(md):
    '''
    Returns the sum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_UB_viol')

    return sum_vm_UB_viol


def sum_vm_LB_viol(md):
    '''
    Returns the sum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_LB_viol')

    return sum_vm_LB_viol


def sum_vm_viol(md):
    '''
    Returns the sum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_viol = get_infeas_from_model_data(md, infeas_name='sum_vm_viol')

    return sum_vm_viol


def sum_thermal_viol(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    sum_thermal_viol = get_infeas_from_model_data(md, infeas_name='sum_thermal_viol')

    return sum_thermal_viol


def avg_vm_UB_viol(md):
    '''
    Returns the average of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_UB_viol')

    return avg_vm_UB_viol

def avg_vm_LB_viol(md):
    '''
    Returns the average of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_LB_viol')

    return avg_vm_LB_viol


def avg_vm_viol(md):
    '''
    Returns the average of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_viol = get_infeas_from_model_data(md, infeas_name='avg_vm_viol')

    return avg_vm_viol


def avg_thermal_viol(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    avg_thermal_viol = get_infeas_from_model_data(md, infeas_name='avg_thermal_viol')

    return avg_thermal_viol


def max_vm_UB_viol(md):
    '''
    Returns the maximum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='max_vm_UB_viol')

    return max_vm_UB_viol


def max_vm_LB_viol(md):
    '''
    Returns the maximum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='max_vm_LB_viol')

    return max_vm_LB_viol


def max_vm_viol(md):
    '''
    Returns the maximum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_viol = get_infeas_from_model_data(md, infeas_name='max_vm_viol')

    return max_vm_viol


def max_thermal_viol(md):
    '''
    Returns the maximum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    max_thermal_viol = get_infeas_from_model_data(md, infeas_name='max_thermal_viol')

    return max_thermal_viol


def pct_vm_UB_viol(md):
    '''
    Returns the number of voltage upper bound infeasibilites
    '''

    if not optimal(md):
        return None
    pct_vm_UB_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_UB_viol')

    return pct_vm_UB_viol


def pct_vm_LB_viol(md):
    '''
    Returns the number of voltage lower bound infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_LB_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_LB_viol')

    return pct_vm_LB_viol


def pct_vm_viol(md):
    '''
    Returns the number of all voltage infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_viol = get_infeas_from_model_data(md, infeas_name='pct_vm_viol')

    return pct_vm_viol


def pct_thermal_viol(md):
    '''
    Returns the number of thermal limit infeasibilites
    '''

    if not optimal(md):
        return None
    pct_thermal_viol = get_infeas_from_model_data(md, infeas_name='pct_thermal_viol')

    return pct_thermal_viol


def acpf_slack(md):
    '''
    Returns the change in the slack bus real power dispatch in the ACPF solution in MW
    '''

    if not optimal(md):
        return None
    acpf_slack = get_infeas_from_model_data(md, infeas_name='acpf_slack')

    return acpf_slack

