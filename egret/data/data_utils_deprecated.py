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
    ## Assumes that if ANY of the sensitivities are present, then the calc_p_sens step can be skipped. This allows us
    ## to skip recalculating the larger test cases that, by default, do not calculate all sensitivities.
    need_branch_sensi = True
    for bn,branch in branches.items():
        if 'ptdf'in branch.keys() and 'pldf' in branch.keys():
            need_branch_sensi = False
            break

    need_bus_sensi = True
    if buses is not None:
        for bn,bus in buses.items():
            if 'ploss_sens' in bus.keys():
                need_bus_sensi = False
                break

    needs_update = need_branch_sensi or need_bus_sensi

    return needs_update


def missing_dense_ptdf_sensitivities(md, branches):
    ## Assumes that if ANY of the sensitivities are present, then the calc_p_sens step can be skipped. This allows us
    ## to skip recalculating the larger test cases that, by default, do not calculate all sensitivities.
    needs_update = True
    for bn,branch in branches.items():
        if 'ptdf' in branch.keys():
            needs_update = False
            break

    return needs_update


def missing_dense_q_sensitivities(md, branches, buses):
    ## Assumes that if ANY of the sensitivities are present, then the calc_q_sens step can be skipped. This allows us
    ## to skip recalculating the larger test cases that, by default, do not calculate all sensitivities.
    need_sys_data = not 'vdf' in md.data['system'].keys() or not 'vdf_c' in md.data['system'].keys()

    need_branch_sensi = True
    for bn,branch in branches.items():
        if 'qtdf' in branch.keys() and 'qldf' in branch.keys():
            need_branch_sensi = False
            break

    need_bus_sensi = False
    for bn,bus in buses.items():
        if 'vdf' in bus.keys() and 'vdf_c' in bus.keys():
            need_bus_sensi = False
            break

    needs_update = need_sys_data or need_branch_sensi or need_bus_sensi

    return needs_update


def missing_sparse_sys_flow_sensitivities(md):

    need_sys_data = not 'Ft' in md.data['system'].keys()
    need_sys_data += not 'ft_c' in md.data['system'].keys()
    need_sys_data += not 'Fv' in md.data['system'].keys()
    need_sys_data += not 'fv_c' in md.data['system'].keys()

    return need_sys_data


def missing_sparse_sys_loss_sensitivities(md):

    need_sys_data = not 'Lt' in md.data['system'].keys()
    need_sys_data += not 'lt_c' in md.data['system'].keys()
    need_sys_data += not 'Lv' in md.data['system'].keys()
    need_sys_data += not 'lv_c' in md.data['system'].keys()

    return need_sys_data


def missing_sparse_branch_p_sensitivities(md,branches):

    need_branch_sensi = False

    for bn,branch in branches.items():
        need_branch_sensi += not 'Ft' in branch.keys()
        need_branch_sensi += not 'ft_c' in branch.keys()
        need_branch_sensi += not 'Lt' in branch.keys()
        need_branch_sensi += not 'lt_c' in branch.keys()

    return need_branch_sensi


def missing_sparse_branch_q_sensitivities(md,branches):

    need_branch_sensi = False

    for bn,branch in branches.items():
        need_branch_sensi += not 'Fv' in branch.keys()
        need_branch_sensi += not 'fv_c' in branch.keys()
        need_branch_sensi += not 'Lv' in branch.keys()
        need_branch_sensi += not 'lv_c' in branch.keys()

    return need_branch_sensi

def create_dicts_of_fdf(md, base_point=BasePointType.SOLUTION):
    create_dicts_of_lccm(md, base_point=base_point)

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']
    file_tag = md.data['system']['model_name']

    update_dense_p = missing_dense_p_sensitivities(md, branches, buses)
    update_dense_q = missing_dense_q_sensitivities(md, branches, buses)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point, filename=file_tag)
        md.data['system']['ptdf'] = p_sens['ptdf']
        md.data['system']['ptdf_c'] = p_sens['ptdf_c']
        #md.data['system']['pldf'] = p_sens['pldf']
        #md.data['system']['pldf_c'] = p_sens['pldf_c']
        md.data['system']['nodal_jacobian_p'] = p_sens['nodal_jacobian_p']
        md.data['system']['offset_jacobian_p'] = p_sens['offset_jacobian_p']

    if update_dense_q:
        q_sens = tx_calc.implicit_calc_q_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point, filename=file_tag)
        md.data['system']['qtdf'] = q_sens['qtdf']
        md.data['system']['qtdf_c'] = q_sens['qtdf_c']
        #md.data['system']['qldf'] = q_sens['qldf']
        #md.data['system']['qldf_c'] = q_sens['qldf_c']
        md.data['system']['vdf'] = q_sens['vdf']
        md.data['system']['vdf_c'] = q_sens['vdf_c']
        md.data['system']['nodal_jacobian_q'] = q_sens['nodal_jacobian_q']
        md.data['system']['offset_jacobian_q'] = q_sens['offset_jacobian_q']

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

    mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(bus_name_list)}
    A = tx_calc.calculate_adjacency_matrix_transpose(branches, branch_name_list, bus_name_list, mapping_bus_to_idx)
    AA = tx_calc.calculate_absolute_adjacency_matrix(A)
    md.data['system']['AdjacencyMat'] = A
    md.data['system']['AbsAdj'] = AA

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
    file_tag = md.data['system']['model_name']

    update_dense_p = missing_dense_p_sensitivities(md, branches, buses)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point, filename=file_tag)
        md.data['system']['ptdf'] = p_sens['ptdf']
        md.data['system']['ptdf_c'] = p_sens['ptdf_c']
        #md.data['system']['pldf'] = p_sens['pldf']
        #md.data['system']['pldf_c'] = p_sens['pldf_c']
        md.data['system']['nodal_jacobian_p'] = p_sens['nodal_jacobian_p']
        md.data['system']['offset_jacobian_p'] = p_sens['offset_jacobian_p']

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


def create_dicts_of_ptdf(md, base_point=BasePointType.FLATSTART, active_branches=None):
    create_dicts_of_lccm(md, base_point=base_point)

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']
    file_tag = md.data['system']['model_name']

    update_dense_p = missing_dense_ptdf_sensitivities(md, branches)

    if update_dense_p:
        p_sens = tx_calc.implicit_calc_p_sens(branches, buses, branch_name_list, bus_name_list, reference_bus, base_point,
                                              active_index_set_branch=active_branches, filename=file_tag)
        md.data['system']['ptdf'] = p_sens['ptdf']
        md.data['system']['ptdf_c'] = p_sens['ptdf_c']
        md.data['system']['nodal_jacobian_p'] = p_sens['nodal_jacobian_p']
        md.data['system']['offset_jacobian_p'] = p_sens['offset_jacobian_p']

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
    sensi = ['Ft', 'ft_c', 'Fv', 'fv_c', 'Lt', 'lt_c', 'Lv', 'lv_c',
             'va_SENSI', 'va_CONST', 'vm_SENSI', 'vm_CONST',
             'ptdf', 'ptdf_c', 'pldf', 'pldf_c', 'qtdf', 'qtdf_c', 'qldf', 'qldf_c', 'vdf', 'vdf_c',
             'ploss_const', 'qloss_const', 'ploss', 'qloss',
             'nodal_jacobian_p', 'offset_jacobian_p', 'nodal_jacobian_q', 'offset_jacobian_q',
             'AdjacencyMat', 'AbsAdj']
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