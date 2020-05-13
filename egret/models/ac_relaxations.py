import pyomo.environ as pe
from .acopf import _create_base_ac_model, create_rsv_acopf_model, create_psv_acopf_model
import egret.model_library.transmission.branch as libbranch
from egret.data.data_utils import map_items, zip_items
from collections import OrderedDict
import egret.model_library.transmission.tx_utils as tx_utils
try:
    import coramin
    coramin_available = True
except ImportError:
    coramin_available = False


def _relaxation_helper(model, md, include_soc, use_linear_relaxation):
    if not coramin_available:
        raise ImportError('Cannot create relaxation unless coramin is available.')
    coramin.relaxations.relax(model,
                              in_place=True,
                              use_fbbt=True,
                              fbbt_options={'deactivate_satisfied_constraints': True,
                                            'max_iter': 2})
    if not use_linear_relaxation:
        for b in coramin.relaxations.relaxation_data_objects(model, descend_into=True, active=True, sort=True):
            if not isinstance(b, coramin.relaxations.PWMcCormickRelaxationData):
                b.use_linear_relaxation = False
                b.rebuild()

    if include_soc:
        branch_attrs = md.attributes(element_type='branch')
        bus_pairs = zip_items(branch_attrs['from_bus'], branch_attrs['to_bus'])
        unique_bus_pairs = list(OrderedDict((val, None) for idx, val in bus_pairs.items()).keys())
        libbranch.declare_ineq_soc(model=model, index_set=unique_bus_pairs,
                                   use_outer_approximation=use_linear_relaxation)


def create_soc_relaxation(model_data, use_linear_relaxation=True):
    model, md = _create_base_ac_model(model_data, include_feasibility_slack=False)
    _relaxation_helper(model=model, md=md, include_soc=True, use_linear_relaxation=use_linear_relaxation)
    return model, md


def create_polar_acopf_relaxation(model_data, include_soc=True, use_linear_relaxation=True):
    model, md = create_psv_acopf_model(model_data, include_feasibility_slack=False)
    _relaxation_helper(model=model, md=md, include_soc=include_soc, use_linear_relaxation=use_linear_relaxation)
    return model, md


def create_rectangular_acopf_relaxation(model_data, include_soc=True, use_linear_relaxation=True):
    model, md = create_rsv_acopf_model(model_data, include_feasibility_slack=False)
    _relaxation_helper(model=model, md=md, include_soc=include_soc, use_linear_relaxation=use_linear_relaxation)
    return model, md


def create_acopf_security_boundary(model_data, include_soc=True, use_linear_relaxation=True):
    model, md = create_psv_acopf_model(model_data, include_feasibility_slack=False)
    _relaxation_helper(model=model, md=md, include_soc=include_soc, use_linear_relaxation=use_linear_relaxation)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    bus_attrs = md.attributes(element_type='bus')
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    buses_with_gens = tx_utils.buses_with_gens(gens)

    expr = 0.
    for bus_name in bus_attrs['names']:
        if bus_name in buses_with_gens:
            expr += (model.vm[bus_name] - pe.value(model.vm[bus_name]))**2
            for gen_name in gens_by_bus[bus_name]:
                expr += (model.pg[gen_name] - pe.value(model.pg[gen_name])) ** 2

    model.del_component('obj')
    model.R = pe.Var(bounds=(0,None))
    model.hypersphere = pe.Constraint(expr = expr <= model.R**2)
    model.obj = pe.Objective(expr = model.R)

    return model, md