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
from egret.common.solver_interface import _solve_model
from pyomo.environ import value

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
    m, md = create_psv_acopf_model(model_data, include_feasibility_slack=True)

    tx_utils.scale_ModelData_to_pu(model_data, inplace=True)

    gens = dict(model_data.elements(element_type='generator'))
    buses = dict(model_data.elements(element_type='bus'))
    branches = dict(model_data.elements(element_type='branch'))
    loads = dict(model_data.elements(element_type='load'))
    shunts = dict(model_data.elements(element_type='shunt'))

    gen_attrs = model_data.attributes(element_type='generator')
    bus_attrs = model_data.attributes(element_type='bus')
    branch_attrs = model_data.attributes(element_type='branch')
    load_attrs = model_data.attributes(element_type='load')
    shunt_attrs = model_data.attributes(element_type='shunt')

    #baseMVA = float(md.data['system']['baseMVA'])

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    # fix variables to the values in modeData object md
    for g, pg in m.pg.items():
        pg.value = gens[g]['pg']
    for g, qg in m.qg.items():
        qg.value = gens[g]['qg']
    for b, va in m.va.items():
        va.value = buses[b]['va']
    for b, vm in m.vm.items():
        vm.value = buses[b]['vm']

    m.pg.fix()
    m.qg.fix()
    #m.va.fix()     ## not sure if it is better to fix these or not
    #m.vm.fix()     ## b/c fixing makes the model (essentially) LP, but leads to infeasibility in some cases

    # remove power flow variable bounds
    for b, pf in m.pf.items():
        pf.setlb(None)
        pf.setub(None)
    for b, pt in m.pt.items():
        pt.setlb(None)
        pt.setub(None)
    for b, qf in m.qf.items():
        qf.setlb(None)
        qf.setub(None)
    for b, qt in m.qt.items():
        qt.setlb(None)
        qt.setub(None)

    # add slack variable to thermal limit constraints
    m.del_component(m.ineq_sf_branch_thermal_limit)
    m.del_component(m.ineq_st_branch_thermal_limit)
    s_thermal_limits = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    slack_init = {k: 0 for k in branch_attrs['names']}
    slack_bounds = {k: (0, s_thermal_limits[k]) for k in branches.keys()}

    decl.declare_var('sf_branch_slack_pos', model=m, index_set=branch_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('st_branch_slack_pos', model=m, index_set=branch_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )

    try:
        con_set = m._con_ineq_s_branch_thermal_limit
    except:
        con_set = decl.declare_set('_con_ineq_s_branch_thermal_limit', model=m, index_set=branch_attrs['names'])

    m.ineq_sf_branch_thermal_limit = pe.Constraint(con_set)
    m.ineq_st_branch_thermal_limit = pe.Constraint(con_set)

    for branch_name in con_set:
        if s_thermal_limits[branch_name] is None:
            continue

        m.ineq_sf_branch_thermal_limit[branch_name] = \
            m.pf[branch_name] ** 2 + m.qf[branch_name] ** 2 \
            <= s_thermal_limits[branch_name] ** 2 + m.sf_branch_slack_pos[branch_name]
        m.ineq_st_branch_thermal_limit[branch_name] = \
            m.pt[branch_name] ** 2 + m.qt[branch_name] ** 2 \
            <= s_thermal_limits[branch_name] ** 2 + m.st_branch_slack_pos[branch_name]

    # calculate infeasibilities
    kcl_p_infeas_expr = sum(m.p_slack_pos[bus_name] + m.p_slack_neg[bus_name] for bus_name in bus_attrs['names'])
    kcl_q_infeas_expr = sum(m.q_slack_pos[bus_name] + m.q_slack_neg[bus_name] for bus_name in bus_attrs['names'])

    thermal_infeas_expr = sum(m.sf_branch_slack_pos[branch_name]
                              + m.st_branch_slack_pos[branch_name]
                              for branch_name in branch_attrs['names'])

    sum_infeas_expr = kcl_p_infeas_expr + kcl_q_infeas_expr + thermal_infeas_expr

    # set objective to sum of infeasibilities (i.e. slacks)
    m.del_component(m.obj)
    m.obj = pe.Objective(expr=sum_infeas_expr)

    # solve model
    print('mult={}'.format(md.data['system']['mult']))
    try:
        m, results = _solve_model(m, "ipopt", timelimit=None, solver_tee=False)
    except:
        print('Solve failed... Increasing slack variable upper bounds.')
        for b, p_slack_pos in m.p_slack_pos.items():
            p_slack_pos.setub(9999)
        for b, p_slack_neg in m.p_slack_neg.items():
            p_slack_neg.setub(9999)
        for b, q_slack_pos in m.q_slack_pos.items():
            q_slack_pos.setub(9999)
        for b, q_slack_neg in m.q_slack_neg.items():
            q_slack_neg.setub(9999)
        for b, sf_branch_slack_pos in m.sf_branch_slack_pos.items():
            sf_branch_slack_pos.setub(9999)
        for b, st_branch_slack_pos in m.st_branch_slack_pos.items():
            st_branch_slack_pos.setub(9999)

        #m.pprint()
        m, results = _solve_model(m, "ipopt", timelimit=None, solver_tee=False)

    show_me = results.Solver.status.key.__str__()
    print('solver status: {}'.format(show_me))

    tx_utils.unscale_ModelData_to_pu(md, inplace=True)

    return m

def get_infeas_from_model_data(md, infeas_name='sum_infeas', overwrite_existing=True):

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

    if not infeas_name in system_data.keys():
        print('...did not find ' + infeas_name + ' in '+ name)
        print(system_data.keys())
    else:
        show_me = pd.DataFrame(system_data,index=[name])
        print('...existing system data: {}'.format(show_me.T))

    # otherwise, solve the sum_infeas model and save solution to md
    m_ac = solve_infeas_model(md)
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
    print('...overwriting system data: {}'.format(show_me.T))

    if 'filename' in system_data.keys():
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
    destination = os.path.join('..','models','tests','transmission_test_instances','approximation_solution_files',model_name)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        print('saving to dest: {}'.format(destination))

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