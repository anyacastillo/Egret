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
import egret.model_library.transmission.tx_calc as tx_calc
from egret.model_library.defn import BasePointType, ApproximationType


def _make_sensi_dict_from_dense( name_list, array ):
    return { n : val for n, val in zip(name_list, array) }
def _make_sensi_dict_from_csr( name_list, csr_array ):
    a_coo = csr_array.tocoo()
    return { name_list[j] : val for j, val in zip(a_coo.col, a_coo.data) }

def missing_dense_p_sensitivities(md, branches, buses=None):

    need_branch_sensi = False
    for branch in branches:
        need_branch_sensi += not hasattr(branch, 'ptdf')
        need_branch_sensi += not hasattr(branch, 'pldf')

    need_bus_sensi = False
    if buses is not None:
        for bus in buses:
            need_branch_sensi += not hasattr(bus, 'ploss_sens')

    needs_update = need_branch_sensi + need_bus_sensi

    return needs_update


def missing_dense_q_sensitivities(md, branches, buses):

    need_sys_data = not hasattr(md.data['system'], 'vm_SENSI') or not hasattr(md.data['system'], 'vm_CONST')

    need_branch_sensi = False
    for branch in branches:
        need_branch_sensi += not hasattr(branch, 'qtdf')
        need_branch_sensi += not hasattr(branch, 'qldf')

    need_bus_sensi = False
    for bus in buses:
        need_branch_sensi += not hasattr(bus, 'vdf')
        need_branch_sensi += not hasattr(bus, 'vdf_c')

    needs_update = need_sys_data or need_branch_sensi or need_bus_sensi

    return needs_update


def missing_sparse_sys_flow_sensitivities(md):

    need_sys_data = not hasattr(md.data['system'], 'Ft')
    need_sys_data += not hasattr(md.data['system'], 'ft_c')
    need_sys_data += not hasattr(md.data['system'], 'Fv')
    need_sys_data += not hasattr(md.data['system'], 'fv_c')

    return need_sys_data


def missing_sparse_sys_loss_sensitivities(md):

    need_sys_data = not hasattr(md.data['system'], 'Lt')
    need_sys_data += not hasattr(md.data['system'], 'lt_c')
    need_sys_data += not hasattr(md.data['system'], 'Lv')
    need_sys_data += not hasattr(md.data['system'], 'lv_c')

    return need_sys_data


def missing_sparse_branch_p_sensitivities(md,branches):

    need_branch_sensi = False

    for branch in branches:
        need_branch_sensi += not hasattr(branch, 'Ft')
        need_branch_sensi += not hasattr(branch, 'ft_c')
        need_branch_sensi += not hasattr(branch, 'Lt')
        need_branch_sensi += not hasattr(branch, 'lt_c')

    return need_branch_sensi


def missing_sparse_branch_q_sensitivities(md,branches):

    need_branch_sensi = False

    for branch in branches:
        need_branch_sensi += not hasattr(branch, 'Fv')
        need_branch_sensi += not hasattr(branch, 'fv_c')
        need_branch_sensi += not hasattr(branch, 'Lv')
        need_branch_sensi += not hasattr(branch, 'lv_c')

    return need_branch_sensi

def missing_ploss_sensitivities(md,buses):

    need_const = not hasattr(md, 'ploss_const')

    need_sens = False
    for bus in buses:
        need_sens += not hasattr(bus, 'ploss_sens')

    needs_update = need_const + need_sens

    return needs_update

def missing_qloss_sensitivities(md,buses):

    need_const = not hasattr(md, 'qloss_const')

    need_sens = False
    for bus in buses:
        need_sens += not hasattr(bus, 'qloss_sens')

    needs_update = need_const + need_sens

    return needs_update

def create_dicts_of_fdf(md, base_point=BasePointType.SOLUTION):
    create_dicts_of_lccm(md, base_point=base_point)

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches, buses)
    update_dense_q = missing_dense_q_sensitivities(md, branches, buses)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = None
        md.data['system']['va_CONST'] = None

    if update_dense_q:
        q_sens = tx_calc.implicit_calc_q_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point)
        md.data['system']['vm_SENSI'] = q_sens['vdf']
        md.data['system']['vm_CONST'] = q_sens['vdf_c']

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, p_sens['ptdf'][idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, p_sens['pldf'][idx])
            branch['ptdf_c'] = p_sens['ptdf_c'][idx]
            branch['pldf_c'] = p_sens['pldf_c'][idx]
            branch['ploss_distribution'] = p_sens['ploss_distribution'][idx]

        if update_dense_q:
            branch['qtdf'] = _make_sensi_dict_from_dense(bus_name_list, q_sens['qtdf'][idx])
            branch['qldf'] = _make_sensi_dict_from_dense(bus_name_list, q_sens['qldf'][idx])
            branch['qtdf_c'] = q_sens['qtdf_c'][idx]
            branch['qldf_c'] = q_sens['qldf_c'][idx]
            branch['qloss_distribution'] = q_sens['qloss_distribution'][idx]


    if update_dense_p:
        md.data['system']['ploss_const'] = p_sens['ploss_const']
        md.data['system']['ploss_resid_const'] = p_sens['ploss_resid_const']
    if update_dense_q:
        md.data['system']['qloss_const'] = q_sens['qloss_const']
        md.data['system']['qloss_resid_const'] = q_sens['qloss_resid_const']

    for idx, bus_name in enumerate(bus_name_list):
        bus = buses[bus_name]
        if update_dense_p:
            bus['ploss_sens'] = p_sens['ploss_sens'][idx]
            bus['ploss_resid_sens'] = p_sens['ploss_resid_sens'][idx]

        if update_dense_q:
            bus['qloss_sens'] = q_sens['qloss_sens'][idx]
            bus['qloss_resid_sens'] = q_sens['qloss_resid_sens'][idx]
            bus['vdf'] = _make_sensi_dict_from_dense(bus_name_list, q_sens['vdf'][idx])
            bus['vdf_c'] = q_sens['vdf_c'][idx]


def create_dicts_of_lccm(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_sparse_flow_sys = missing_sparse_sys_flow_sensitivities(md)
    update_sparse_loss_sys = missing_sparse_sys_loss_sensitivities(md)
    update_sparse_branch_p = missing_sparse_branch_p_sensitivities(md, branches)
    update_sparse_branch_q = missing_sparse_branch_q_sensitivities(md, branches)

    if update_sparse_flow_sys:
        Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Ft'] = Ft
        md.data['system']['ft_c'] = ft_c
        md.data['system']['Fv'] = Fv
        md.data['system']['fv_c'] = fv_c

    if update_sparse_loss_sys:
        Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Lt'] = Lt
        md.data['system']['lt_c'] = lt_c
        md.data['system']['Lv'] = Lv
        md.data['system']['lv_c'] = lv_c

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_sparse_branch_p:
            branch['Ft'] = _make_sensi_dict_from_csr(bus_name_list, Ft[idx])
            branch['Lt'] = _make_sensi_dict_from_csr(bus_name_list, Lt[idx])
            branch['ft_c'] = ft_c[idx]
            branch['lt_c'] = lt_c[idx]

        if update_sparse_branch_q:
            branch['Fv'] = _make_sensi_dict_from_csr(bus_name_list, Fv[idx])
            branch['Lv'] = _make_sensi_dict_from_csr(bus_name_list, Lv[idx])
            branch['fv_c'] = fv_c[idx]
            branch['lv_c'] = lv_c[idx]


def create_dicts_of_fdf_simplified(md, base_point=BasePointType.SOLUTION):
    create_dicts_of_fdf(md,base_point)


def create_dicts_of_ptdf_losses(md, base_point=BasePointType.SOLUTION):
    create_dicts_of_lccm(md, base_point=base_point)

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches, buses)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = None
        md.data['system']['va_CONST'] = None

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, p_sens['ptdf'][idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, p_sens['pldf'][idx])
            branch['ptdf_c'] = p_sens['ptdf_c'][idx]
            branch['pldf_c'] = p_sens['pldf_c'][idx]
            branch['ploss_distribution'] = p_sens['ploss_distribution'][idx]


    if update_dense_p:
        md.data['system']['ploss_const'] = p_sens['ploss_const']

    for idx, bus_name in enumerate(bus_name_list):
        bus = buses[bus_name]
        if update_dense_p:
            bus['ploss_sens'] = p_sens['ploss_sens'][idx]


def create_dicts_of_ptdf(md, base_point=BasePointType.FLATSTART):
    create_dicts_of_lccm(md, base_point=base_point)

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches, buses)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = None
        md.data['system']['va_CONST'] = None

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, p_sens['ptdf'][idx])
            branch['ptdf_c'] = p_sens['ptdf_c'][idx]



def destroy_dicts_of_fdf(md):

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))

    # delete sensitivity matrices from 'system'. May need to add these back to modelData when opening the .json file.
    sensi = ['Ft', 'ft_c', 'Fv', 'fv_c', 'Lt', 'lt_c', 'Lv', 'lv_c', 'va_SENSI', 'va_CONST', 'vm_SENSI', 'vm_CONST','ploss_const','qloss_const','ploss','qloss']
    for s in sensi:
        if s in md.data['system']:
            del md.data['system'][s]

    # delete sensitivities from 'branch'
    sensi = ['ptdf', 'pldf', 'qtdf', 'qldf', 'ptdf_c', 'pldf_c', 'qtdf_c', 'qldf_c','ploss_distribution','qloss_distribution']
    for branch_name, branch in branches.items():
        for s in sensi:
            if s in branch:
                del branch[s]

    # delete sensitivities from 'bus'
    sensi = ['vdf', 'vdf_c', 'phi_from', 'phi_to', 'phi_loss_from', 'phi_loss_to',
             'phi_q_from', 'phi_q_to', 'phi_loss_q_from', 'phi_loss_q_to','ploss_sens','qloss_sens']
    for bus_name, bus in buses.items():
        for s in sensi:
            if s in bus:
                del bus[s]

    return

def destroy_solution_data_of_md(md):
    del md.elements
    del md.data['system']['total_cost']