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
import egret.model_library.decl as decl
import pyomo.environ as pe
import pandas as pd
from egret.models.acopf import create_psv_acopf_model
from egret.models.acpf import create_psv_acpf_model, solve_acpf
from egret.common.solver_interface import _solve_model
from pyomo.environ import value
import egret.data.data_utils_deprecated as data_utils_deprecated
from math import sqrt

def solve_time(md):

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

    val = md.data['results']['#_cons']

    return val


def num_variables(md):

    val = md.data['results']['#_vars']

    return val


def num_nonzeros(md):

    results = md.data['results']

    if hasattr(results, '#_nz'):
        val = results['#_nz']
        return val

    return None

def model_sparsity(md):

    results = md.data['results']

    try:
        nc = results['#_cons']
        nv = results['#_vars']
        nz = results['#_nz']
        val = nz / ( nc * nv )
        return val
    except:
        return None


def total_cost(md):

    val = md.data['system']['total_cost']

    return val

def ploss(md):

    val = md.data['system']['ploss']

    return val

def qloss(md):

    val = md.data['system']['qloss']

    return val

def pgen(md):

    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['pg']

    return dispatch

def qgen(md):

    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['qg']

    return dispatch

def pflow(md):

    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['pf']

    return flow

def qflow(md):

    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['qf']

    return flow

def vmag(md):

    buses = dict(md.elements(element_type='bus'))
    vm = {}

    for b,bus in buses.items():
        vm[b] = bus['vm']

    return vm


def solve_infeas_model(model_data):


    # build ACOPF model with fixed gen output, fixed voltage angle/mag, and relaxed power balance
    md, m, results = solve_acpf(model_data, "ipopt", return_results=True, return_model=True, solver_tee=False)

    sum_infeas_expr= 0. # MW + MVAr + MVA
    m.kcl_p_infeas_expr = 0. # MW
    m.kcl_q_infeas_expr = 0. # MVAr
    m.thermal_infeas_expr = 0. #MVA
    # sum_infeas_expr = kcl_p_infeas_expr + kcl_q_infeas_expr + thermal_infeas_expr

    # set objective to sum of infeasibilities (i.e. slacks)
    m.del_component(m.obj)
    m.obj = pe.Objective(expr=sum_infeas_expr)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    m.p_slack_pos = dict()
    m.p_slack_neg = dict()
    m.q_slack_pos = dict()
    m.q_slack_neg = dict()
    ref_bus = md.data['system']['reference_bus']
    for bus_name, bus_dict in buses.items():
        m.p_slack_pos[bus_name] = 0.
        m.p_slack_neg[bus_name] = 0.
        m.q_slack_pos[bus_name] = 0.
        m.q_slack_neg[bus_name] = 0.
        for gen_name in gens_by_bus[bus_name]:
            g_dict = gens[gen_name]
            if bus_name == ref_bus:
                if value(m.pg[gen_name]) > g_dict['p_max']:
                    m.p_slack_pos[bus_name] += value(m.pg[gen_name]) - g_dict['p_max']
                if value(m.pg[gen_name]) < g_dict['p_min']:
                    m.p_slack_neg[bus_name] += g_dict['p_min']-value(m.pg[gen_name])
                m.kcl_p_infeas_expr += m.p_slack_pos[bus_name] + m.p_slack_neg[bus_name]
            if value(m.qg[gen_name]) > g_dict['q_max']:
                m.q_slack_pos[bus_name] += value(m.qg[gen_name]) - g_dict['q_max']
            if value(m.qg[gen_name]) < g_dict['q_min']:
                m.q_slack_neg[bus_name] += g_dict['q_min'] - value(m.qg[gen_name])
            m.kcl_q_infeas_expr += m.q_slack_pos[bus_name] + m.q_slack_neg[bus_name]

    m.sf_branch_slack_pos = dict()
    m.st_branch_slack_pos = dict()
    for branch_name, branch_dict in branches.items():
        m.sf_branch_slack_pos[branch_name] = 0.
        m.st_branch_slack_pos[branch_name] = 0.
        sf = sqrt(branch_dict["pf"]**2 + branch_dict["qf"]**2)
        st = sqrt(branch_dict["pt"]**2 + branch_dict["qt"]**2)

        if sf > branch_dict['rating_long_term']:
            m.sf_branch_slack_pos[branch_name] = sf - branch_dict['rating_long_term']
            m.thermal_infeas_expr += m.sf_branch_slack_pos[branch_name]
        if st > branch_dict['rating_long_term']:
            m.st_branch_slack_pos[branch_name] = st - branch_dict['rating_long_term']
            m.thermal_infeas_expr += m.st_branch_slack_pos[branch_name]

    sum_infeas_expr = m.kcl_p_infeas_expr + m.kcl_q_infeas_expr + m.thermal_infeas_expr
    m.obj = pe.Objective(expr=sum_infeas_expr)
    return m, results

def get_infeas_from_model_data(md, infeas_name='sum_infeas', overwrite_existing=False):

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
    m_ac, results = solve_infeas_model(md)
    termination = results.solver.termination_condition.__str__()
    message += 'Infeasibility model returned ' + termination + '.'
    print(message)
    #m_ac.pprint()

    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    kcl_p_list = [value(m_ac.p_slack_pos[bus_name]) + value(m_ac.p_slack_neg[bus_name])
                  for bus_name in bus_attrs['names']]

    kcl_q_list = [value(m_ac.q_slack_pos[bus_name]) + value(m_ac.q_slack_neg[bus_name])
                  for bus_name in bus_attrs['names']]

    thermal_list = [value(m_ac.sf_branch_slack_pos[branch_name]) + value(m_ac.st_branch_slack_pos[branch_name])
                    for branch_name in branch_attrs['names']]

    kcl_p_infeas = sum(kcl_p_list)
    kcl_q_infeas = sum(kcl_q_list)
    thermal_infeas = sum(thermal_list)

    system_data['kcl_p_infeas'] = kcl_p_infeas
    system_data['avg_kcl_p_infeas'] = kcl_p_infeas / len(kcl_p_list)
    system_data['max_kcl_p_infeas'] = max(kcl_p_list)

    system_data['kcl_q_infeas'] = kcl_q_infeas
    system_data['avg_kcl_q_infeas'] = kcl_q_infeas / len(kcl_q_list)
    system_data['max_kcl_q_infeas'] = max(kcl_q_list)

    system_data['thermal_infeas'] = thermal_infeas
    system_data['avg_thermal_infeas'] = thermal_infeas / len(thermal_list)
    system_data['max_thermal_infeas'] = max(thermal_list)

    system_data['sum_infeas'] = kcl_p_infeas + kcl_q_infeas + thermal_infeas

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
    current_dir, current_file = os.path.split(os.path.realpath(__file__))

    # move to case directory
    source = os.path.join(cwd, filename + '.json')
    destination = os.path.join(cwd,'transmission_test_instances','approximation_solution_files',model_name)

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


def kcl_p_infeas(md):
    '''
    Returns sum of real power balance infeasibilites (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    kcl_p_infeas = get_infeas_from_model_data(md, infeas_name='kcl_p_infeas')

    return kcl_p_infeas

def kcl_q_infeas(md):
    '''
    Returns sum of reactive power balance infeasibilities (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    kcl_q_infeas = get_infeas_from_model_data(md, infeas_name='kcl_q_infeas')

    return kcl_q_infeas


def thermal_infeas(md):
    '''
    Returns sum of thermal limit infeasibilities (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    thermal_infeas = get_infeas_from_model_data(md, infeas_name='thermal_infeas')

    return thermal_infeas


def avg_kcl_p_infeas(md):
    '''
    Returns sum of real power balance infeasibilites (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    kcl_p_infeas = get_infeas_from_model_data(md, infeas_name='avg_kcl_p_infeas')

    return kcl_p_infeas

def avg_kcl_q_infeas(md):
    '''
    Returns sum of reactive power balance infeasibilities (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    kcl_q_infeas = get_infeas_from_model_data(md, infeas_name='avg_kcl_q_infeas')

    return kcl_q_infeas


def avg_thermal_infeas(md):
    '''
    Returns sum of thermal limit infeasibilities (i.e., slack variables)
    Note: returned value is in p.u.
    '''

    thermal_infeas = get_infeas_from_model_data(md, infeas_name='avg_thermal_infeas')

    return thermal_infeas


def max_kcl_p_infeas(md):
    '''
    Returns the largest real power balance violation (i.e., slack variable)
    Note: returned value is in p.u.
    '''

    max_kcl_p_infeas = get_infeas_from_model_data(md, infeas_name='max_kcl_p_infeas')

    return max_kcl_p_infeas


def max_kcl_q_infeas(md):
    '''
    Returns the largest reactive power balance violation (i.e., slack variable)
    Note: returned value is in p.u.
    '''

    max_kcl_q_infeas = get_infeas_from_model_data(md, infeas_name='max_kcl_q_infeas')

    return max_kcl_q_infeas


def max_thermal_infeas(md):
    '''
    Returns slargest thermal limit violation (i.e., slack variable)
    Note: returned value is in p.u.
    '''

    max_thermal_infeas = get_infeas_from_model_data(md, infeas_name='max_thermal_infeas')

    return max_thermal_infeas

def sum_infeas(md):
    '''
    Returns sum of all infeasibilites (i.e., power balance and thermal limit slacks)
    Note: returned value is in p.u.
    '''

    sum_infeas = get_infeas_from_model_data(md, infeas_name='sum_infeas')

    return sum_infeas