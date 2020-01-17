#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This file includes the solver interfaces for EGRET.
"""
import pyomo.opt as po
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver


def _set_options(solver, mipgap=None, timelimit=None, other_options=None):
    '''
    Create options

    Parameters
    ----------
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instanciated pyomo solver
    mipgap : float (optional)
        Mipgap to use for unit commitment solve; default is 0.001
    timelimit : float (optional)
        Time limit for unit commitment run. Default of None results in no time
        limit being set -- runs until mipgap is satisfied
    other_options : dict (optional)
        Other options to pass into the solver. Default is dict().
    uc_model_generator : function (optional)
        Function for generating the unit commitment model. Default is
        egret.models.unit_commitment.create_tight_unit_commitment_model
    relaxed : bool (optional)
        If True, creates a relaxed unit commitment model
    return_model : bool (optional)
        If True, returns the pyomo model object
    '''

    solver_name = solver.name

    if 'gurobi' in solver_name:
        if mipgap is not None:
            solver.options.MIPGap = mipgap
        if timelimit is not None:
            solver.options.TimeLimit = timelimit
    elif 'cplex' in solver_name:
        if mipgap is not None:
            solver.options.mip_tolerances_mipgap = mipgap
        if timelimit is not None:
            solver.options.timelimit = timelimit
    elif 'glpk' in solver_name:
        if mipgap is not None:
            solver.options.mipgap = mipgap
        if timelimit is not None:
            solver.options.tmlim = timelimit
    elif 'cbc' in solver_name:
        if mipgap is not None:
            solver.options.ratioGap = mipgap
        if timelimit is not None:
            solver.options.sec = timelimit
    # else:
    #     raise Exception('Solver {0} not recognized'.format(solver_name))

    if other_options is not None:
        for key, opt in other_options.items():
            solver.options[key] = opt

def _solve_model(model,
                 solver,
                 mipgap=0.001,
                 timelimit = None,
                 solver_tee = True,
                 symbolic_solver_labels = False,
                 options = None,
                 return_solver = False,
                 vars_to_load = None):
    '''
    Create and solve an Egret power system optimization model

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        A pyomo ConcreteModel object.
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instanciated pyomo solver
    mipgap : float (optional)
        Mipgap to use for unit commitment solve; default is 0.001
    timelimit : float (optional)
        Time limit for unit commitment run. Default of None results in no time
        limit being set -- runs until mipgap is satisfied
    solver_tee : bool (optional)
        Display solver log. Default is True.
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels. Useful for debugging; default is False.
    options : dict (optional)
        Other options to pass into the solver. Default is dict().
    return_solver : bool (optional)
        Returns the solver object
    vars_to_load : list (optional)
        When supplied, and the solver is persistent, this will just load
        pyomo variables specificed

    Returns
    -------

    '''

    results = None

    ## termination conditions which are acceptable
    safe_termination_conditions = [
                                   po.TerminationCondition.maxTimeLimit,
                                   po.TerminationCondition.maxIterations,
                                   po.TerminationCondition.minFunctionValue,
                                   po.TerminationCondition.minStepLength,
                                   po.TerminationCondition.globallyOptimal,
                                   po.TerminationCondition.locallyOptimal,
                                   po.TerminationCondition.feasible,
                                   po.TerminationCondition.optimal,
                                   po.TerminationCondition.maxEvaluations,
                                   po.TerminationCondition.other,
                                  ]

    if isinstance(solver, str):
        solver = po.SolverFactory(solver)
    elif isinstance(solver, po.base.OptSolver):
        pass
    else:
        raise Exception('solver must be string or an instantiated pyomo solver')

    _set_options(solver, mipgap, timelimit, options)

    if isinstance(solver, PersistentSolver):
        solver.set_instance(model, symbolic_solver_labels=symbolic_solver_labels)
        results = solver.solve(model, tee=solver_tee, load_solutions=False, save_results=False)
    else:
        results = solver.solve(model, tee=solver_tee, \
                              symbolic_solver_labels=symbolic_solver_labels, load_solutions=False)

    flag = True
    if results.solver.termination_condition not in safe_termination_conditions:
        if model._ptdf_options["lazy"]:
            return model, results, solver
        raise Exception('Problem encountered during solve, termination_condition {}'.format(results.solver.termination_condition))

    if isinstance(solver, PersistentSolver):
        solver.load_vars(vars_to_load)
        if vars_to_load is None:
            if hasattr(model, "dual"):
                solver.load_duals()
            if hasattr(model, "slack"):
                solver.load_slacks()
    else:
        model.solutions.load_from(results)

    if return_solver:
        return model, results, solver
    return model, results
