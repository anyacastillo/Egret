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

def missing_dense_p_sensitivities(md, branches):

    need_sys_data = not hasattr(md.data['system'], 'va_SENSI') or not hasattr(md.data['system'], 'va_CONST')

    need_branch_sensi = False
    for branch in branches:
        need_branch_sensi += not hasattr(branch, 'ptdf')
        need_branch_sensi += not hasattr(branch, 'pldf')

    needs_update = need_sys_data or need_branch_sensi

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
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches)
    update_dense_q = missing_dense_q_sensitivities(md, branches, buses)
    update_sparse_flow = missing_sparse_sys_flow_sensitivities(md)
    update_sparse_loss = missing_sparse_sys_loss_sensitivities(md)

    if update_dense_p:
        ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = va_sensi
        md.data['system']['va_CONST'] = va_const

    if update_dense_q:
        qtdf, qtdf_c, qldf, qldf_c, vm_sensi, vm_const = tx_calc.calculate_qtdf_qldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['vm_SENSI'] = vm_sensi
        md.data['system']['vm_CONST'] = vm_const

    if update_sparse_flow:
        Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Ft'] = Ft
        md.data['system']['ft_c'] = ft_c
        md.data['system']['Fv'] = Fv
        md.data['system']['fv_c'] = fv_c

    if update_sparse_loss:
        Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Lt'] = Lt
        md.data['system']['lt_c'] = lt_c
        md.data['system']['Lv'] = Lv
        md.data['system']['lv_c'] = lv_c

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']
    total_ploss = md.data['system']['ploss']
    total_qloss = md.data['system']['qloss']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, ptdf[idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, pldf[idx])
            branch['ptdf_c'] = ptdf_c[idx]
            branch['pldf_c'] = pldf_c[idx]

        if update_dense_q:
            branch['qtdf'] = _make_sensi_dict_from_dense(bus_name_list, qtdf[idx])
            branch['qldf'] = _make_sensi_dict_from_dense(bus_name_list, qldf[idx])
            branch['qtdf_c'] = qtdf_c[idx]
            branch['qldf_c'] = qldf_c[idx]


        ## loss distributions get updated every time (this could cause some inconsistency)
        if total_ploss > 0:
            branch['ploss_distribution'] = (branch['pf'] + branch['pt']) / total_ploss
        else:
            branch['ploss_distribution'] = 0
        if total_qloss > 0:
            branch['qloss_distribution'] = (branch['qf'] + branch['qt']) / total_qloss
        else:
            branch['qloss_distribution'] = 0


    # TODO: check if phi-constants are used anywhere
    #phi_from, phi_to = tx_calc.calculate_phi_constant(branches, branch_attrs['names'], bus_attrs['names'],
    #                                                  ApproximationType.PTDF_LOSSES)
    #phi_loss_from, phi_loss_to = tx_calc.calculate_phi_loss_constant(branches, branch_attrs['names'],
    #                                                                 bus_attrs['names'], ApproximationType.PTDF_LOSSES)
    #phi_q_from, phi_q_to = tx_calc.calculate_phi_q_constant(branches, branch_attrs['names'], bus_attrs['names'])
    #phi_loss_q_from, phi_loss_q_to = tx_calc.calculate_phi_loss_q_constant(branches, branch_attrs['names'], bus_attrs['names'])


    if update_dense_q:

        for idx, bus_name in enumerate(bus_name_list):

            bus = buses[bus_name]
            bus['vdf'] = _make_sensi_dict_from_dense(bus_name_list, vm_sensi[idx])
            bus['vdf_c'] = vm_const[idx]

            #bus['phi_from'] = _make_sensi_dict_from_csr(branch_name_list, phi_from[idx])
            #bus['phi_to'] = _make_sensi_dict_from_csr(branch_name_list, phi_to[idx])
            #bus['phi_loss_from'] = _make_sensi_dict_from_csr(branch_name_list, phi_loss_from[idx])
            #bus['phi_loss_to'] = _make_sensi_dict_from_csr(branch_name_list, phi_loss_to[idx])
            #bus['phi_q_from'] = _make_sensi_dict_from_csr(branch_name_list, phi_q_from[idx])
            #bus['phi_q_to'] = _make_sensi_dict_from_csr(branch_name_list, phi_q_to[idx])
            #bus['phi_loss_q_from'] = _make_sensi_dict_from_csr(branch_name_list, phi_loss_q_from[idx])
            #bus['phi_loss_q_to'] = _make_sensi_dict_from_csr(branch_name_list, phi_loss_q_to[idx])

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


    #_len_branch = len(branch_attrs['names'])
    #_mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    #_len_bus = len(bus_attrs['names'])
    #_mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    #for idx, branch_name in _mapping_branch.items():
    #    branch = md.data['elements']['branch'][branch_name]
    #    _row_Ft = {bus_attrs['names'][i]: Ft[idx, i] for i in list(range(0, _len_bus))}
    #    branch['Ft'] = _row_Ft
    #    _row_Lt = {bus_attrs['names'][i]: Lt[idx, i] for i in list(range(0, _len_bus))}
    #    branch['Lt'] = _row_Lt
    #    branch['ft_c'] = ft_c[idx]
    #    branch['lt_c'] = lt_c[idx]
    #    _row_Fv = {bus_attrs['names'][i]: Fv[idx, i] for i in list(range(0, _len_bus))}
    #    branch['Fv'] = _row_Fv
    #    _row_Lv = {bus_attrs['names'][i]: Lv[idx, i] for i in list(range(0, _len_bus))}
    #    branch['Lv'] = _row_Lv
    #    branch['fv_c'] = fv_c[idx]
    #    branch['lv_c'] = lv_c[idx]


def create_dicts_of_fdf_simplified(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches)
    update_dense_q = missing_dense_q_sensitivities(md, branches, buses)
    update_ploss_sens = missing_ploss_sensitivities(md, buses)
    update_qloss_sens = missing_qloss_sensitivities(md, buses)
    update_sparse_flow = missing_sparse_sys_flow_sensitivities(md)
    update_sparse_loss = missing_sparse_sys_loss_sensitivities(md)

    if update_dense_p:
        ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = va_sensi
        md.data['system']['va_CONST'] = va_const

    if update_dense_q:
        qtdf, qtdf_c, qldf, qldf_c, vm_sensi, vm_const = tx_calc.calculate_qtdf_qldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['vm_SENSI'] = vm_sensi
        md.data['system']['vm_CONST'] = vm_const

    if update_sparse_flow:
        Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Ft'] = Ft
        md.data['system']['ft_c'] = ft_c
        md.data['system']['Fv'] = Fv
        md.data['system']['fv_c'] = fv_c

    if update_sparse_loss:
        Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Lt'] = Lt
        md.data['system']['lt_c'] = lt_c
        md.data['system']['Lv'] = Lv
        md.data['system']['lv_c'] = lv_c

    branches = md.data['elements']['branch']
    buses = md.data['elements']['bus']
    total_ploss = md.data['system']['ploss']
    total_qloss = md.data['system']['qloss']

    for idx, branch_name in enumerate(branch_name_list):
        branch = branches[branch_name]

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, ptdf[idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, pldf[idx])
            branch['ptdf_c'] = ptdf_c[idx]
            branch['pldf_c'] = pldf_c[idx]

        if update_dense_q:
            branch['qtdf'] = _make_sensi_dict_from_dense(bus_name_list, qtdf[idx])
            branch['qldf'] = _make_sensi_dict_from_dense(bus_name_list, qldf[idx])
            branch['qtdf_c'] = qtdf_c[idx]
            branch['qldf_c'] = qldf_c[idx]


        ## loss distributions get updated every time (this could cause some inconsistency)
        if total_ploss > 0:
            branch['ploss_distribution'] = (branch['pf'] + branch['pt']) / total_ploss
        else:
            branch['ploss_distribution'] = 0
        if total_qloss > 0:
            branch['qloss_distribution'] = (branch['qf'] + branch['qt']) / total_qloss
        else:
            branch['qloss_distribution'] = 0


    # need to sum over branches.items() since pldf_c may not have been independently calculated
    if update_ploss_sens:
        md.data['system']['ploss_const'] = sum(branch['pldf_c'] for bn,branch in branches.items())
    if update_qloss_sens:
        md.data['system']['qloss_const'] = sum(branch['qldf_c'] for bn,branch in branches.items())

    for idx, bus_name in enumerate(bus_name_list):
        bus = buses[bus_name]

        # condensed loss factors are pldf and qldf summed over the set of branches
        if update_ploss_sens:
            bus['ploss_sens'] = sum(branch['pldf'][bus_name] for bn,branch in branches.items())

        if update_qloss_sens:
            bus['qloss_sens'] = sum(branch['qldf'][bus_name] for bn,branch in branches.items())

        # voltage distribution factors are the same as in FDF
        if update_dense_q:
            bus['vdf'] = _make_sensi_dict_from_dense(bus_name_list, vm_sensi[idx])
            bus['vdf_c'] = vm_const[idx]



def create_dicts_of_ptdf_losses(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches)

    update_ploss_sens = missing_ploss_sensitivities(md, buses)

    update_sparse_flow = missing_sparse_sys_flow_sensitivities(md)
    update_sparse_loss = missing_sparse_sys_loss_sensitivities(md)

    if update_dense_p:
        ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = va_sensi
        md.data['system']['va_CONST'] = va_const

    if update_sparse_flow:
        Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Ft'] = Ft
        md.data['system']['ft_c'] = ft_c
        md.data['system']['Fv'] = Fv
        md.data['system']['fv_c'] = fv_c

    if update_sparse_loss:
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

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, ptdf[idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, pldf[idx])
            branch['ptdf_c'] = ptdf_c[idx]
            branch['pldf_c'] = pldf_c[idx]

    # need to sum over branches.items() since pldf_c may not have been independently calculated
    if update_ploss_sens:
        md.data['system']['ploss_const'] = sum(branch['pldf_c'] for bn,branch in branches.items())

        for idx, bus_name in enumerate(bus_name_list):
            bus = buses[bus_name]

            # condensed loss factors are pldf summed over the set of branches
            bus['ploss_sens'] = sum(branch['pldf'][bus_name] for bn,branch in branches.items())


def create_dicts_of_ptdf(md, base_point=BasePointType.FLATSTART):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    branch_name_list = branch_attrs['names']
    bus_name_list = bus_attrs['names']

    reference_bus = md.data['system']['reference_bus']

    update_dense_p = missing_dense_p_sensitivities(md, branches)

    update_sparse_flow = missing_sparse_sys_flow_sensitivities(md)
    update_sparse_loss = missing_sparse_sys_loss_sensitivities(md)

    if update_dense_p:
        ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_name_list,
                                                    bus_name_list, reference_bus, base_point)
        md.data['system']['va_SENSI'] = va_sensi
        md.data['system']['va_CONST'] = va_const

    if update_sparse_flow:
        Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_name_list, bus_name_list,
                                    reference_bus, base_point)
        md.data['system']['Ft'] = Ft
        md.data['system']['ft_c'] = ft_c
        md.data['system']['Fv'] = Fv
        md.data['system']['fv_c'] = fv_c

    if update_sparse_loss:
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

        if update_dense_p:
            branch['ptdf'] = _make_sensi_dict_from_dense(bus_name_list, ptdf[idx])
            branch['pldf'] = _make_sensi_dict_from_dense(bus_name_list, pldf[idx])
            branch['ptdf_c'] = ptdf_c[idx]
            branch['pldf_c'] = pldf_c[idx]



def destroy_dicts_of_fdf(md):

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))

    # delete sensitivity matrices from 'system'. May need to add these back to modelData when opening the .json file.
    sensi = ['Ft', 'ft_c', 'Fv', 'fv_c', 'Lt', 'lt_c', 'Lv', 'lv_c', 'va_SENSI', 'va_CONST', 'vm_SENSI', 'vm_CONST']
    for s in sensi:
        if s in md.data['system']:
            del md.data['system'][s]

    # delete sensitivities from 'branch'
    sensi = ['ptdf', 'pldf', 'qtdf', 'qldf', 'ptdf_c', 'pldf_c', 'qtdf_c', 'qldf_c',]
    for branch in branches:
        for s in sensi:
            if s in branch:
                del branch[s]


    # delete sensitivities from 'bus'
    sensi = ['vdf', 'vdf_c', 'phi_from', 'phi_to', 'phi_loss_from', 'phi_loss_to',
             'phi_q_from', 'phi_q_to', 'phi_loss_q_from', 'phi_loss_q_to']
    for bus in buses:
        for s in sensi:
            if s in bus:
                del bus[s]


