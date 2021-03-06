#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## file for ramping constraints
from pyomo.environ import *
import math

from .uc_utils import add_model_attr 
component_name = 'ramping_limits'

generation_limits_w_startup_shutdown = ['MLR_generation_limits',
                                        'gentile_generation_limits',
                                        'pan_guan_gentile_generation_limits',
                                        'pan_guan_gentile_KOW_generation_limits',
                                        ]

def _damcikurt_basic_ramping(model):

    ## NOTE: with the expression MaximumPowerAvailableAboveMinimum and PowerGeneratedAboveMinimum, 
    ##       these constraints are expressed as needed, there's no cancelation even though we end
    ##       up using these expressions
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
          # if the unit was on in t0, then it's m.PowerGeneratedT0[g] >= m.MinimumPowerOutput[g], and m.UnitOnT0 == 1
          # if not, then m.UnitOnT0[g] == 0 and so (m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]) * m.UnitOnT0[g] is 0
            return m.MaximumPowerAvailableAboveMinimum[g, t] <= m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] + \
                                                  m.ScaledNominalRampUpLimit[g]*m.UnitOn[g,t] + \
    					      (m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g]- m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t] 
        else:
            return m.MaximumPowerAvailableAboveMinimum[g, t] <= m.PowerGeneratedAboveMinimum[g, t-1] + m.ScaledNominalRampUpLimit[g]*m.UnitOn[g,t] + \
    					      			(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g] - m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t] 
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    
    def enforce_ramp_down_limits_rule(m, g, t):
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                return m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g]*m.UnitOnT0[g] + (m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g] - m.ScaledNominalRampDownLimit[g])*m.UnitStop[g,t]
        else:
            return m.PowerGeneratedAboveMinimum[g, t-1] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g]*m.UnitOn[g,t-1] + (m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g] - m.ScaledNominalRampDownLimit[g])*m.UnitStop[g,t]
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

    return

@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': ['garver_3bin_vars','garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars'],
                                            'power_vars': None, 
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def damcikurt_ramping(model):
    '''
    Equations (3) and (18) from

    Pelin Damci-Kurt, Simge Kucukyavuz, Deepak Rajan, and Alper Atamturk. A
    polyhedral study of production ramping. Mathematical Programming,
    158(1-2):175–205, 2016.
    '''
    _damcikurt_basic_ramping(model)


@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': ['garver_3bin_vars','garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars'],
                                            'power_vars': None, 
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def damcikurt_ramping_2period(model):
    '''
    Equations (3) and (18), plus equations (20) and (23), from

    Pelin Damci-Kurt, Simge Kucukyavuz, Deepak Rajan, and Alper Atamturk. A
    polyhedral study of production ramping. Mathematical Programming,
    158(1-2):175–205, 2016.
    '''
    _damcikurt_basic_ramping(model)

    def two_period_ramp_up_rule(m, g, t):
        if value(m.ScaledStartupRampLimit[g]) < value(m.MinimumPowerOutput[g] + m.ScaledNominalRampUpLimit[g]):
            return Constraint.Skip
        j = math.floor(min(value(m.NumTimePeriods)-t, value(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g])/value(m.ScaledNominalRampUpLimit[g])))
        if j > 1: ## j == 1 is handled above
            return m.MaximumPowerAvailableAboveMinimum[g,t+j] - m.PowerGeneratedAboveMinimum[g,t] <= j*m.ScaledNominalRampUpLimit[g]*m.UnitOn[g,t+j] \
                    + sum( min(value(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g] - i*m.ScaledNominalRampUpLimit[g]), \
                                value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g] - j*m.ScaledNominalRampUpLimit[g]))*m.UnitStart[g,i] for i in range(1, j+1) ) 
        return Constraint.Skip
    model.EnforceTwoPeriodRampUpRule = Constraint(model.ThermalGenerators, model.TimePeriods, rule=two_period_ramp_up_rule)

    def two_period_ramp_down_rule(m, g, t):
        if value(m.ScaledShutdownRampLimit[g]) < value(m.MinimumPowerOutput[g] + m.ScaledNominalRampDownLimit[g]):
            return Constraint.Skip
        j = math.floor(min(value(m.NumTimePeriods)-t, value(m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g])/value(m.ScaledNominalRampDownLimit[g])))
        if j > 1: ## j == 1 is handled above
            return m.PowerGeneratedAboveMinimum[g,t] - m.PowerGeneratedAboveMinimum[g,t+j] <= j*m.ScaledNominalRampDownLimit[g]*m.UnitOn[g,t] \
                    + sum( min(value(m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g] - (j-i+1)*m.ScaledNominalRampDownLimit[g]), \
                                value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g] - j*m.ScaledNominalRampDownLimit[g]))*m.UnitStop[g,t+i] for i in range(1, j+1) ) 
        return Constraint.Skip
    model.EnforceTwoPeriodRampDownRule = Constraint(model.ThermalGenerators, model.TimePeriods, rule=two_period_ramp_down_rule)


    
@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': ['ALS_state_transition_vars'],
                                            'power_vars': None, 
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def ALS_damcikurt_ramping(model):
    '''
    Equations (20a) and (20b) from
    
    Semih Atakan, Guglielmo Lulli, and Suvrajeet Sen. A state transition MIP
    formulation for the unit commitment problem. IEEE Transactions on Power
    Systems, 33(1):736–748, 2018.

    which are modifications of the damcikurt ramping limits.
    '''

    ## NOTE: with the expression MaximumPowerAvailableAboveMinimum and PowerGeneratedAboveMinimum, 
    ##       these constraints are expressed as needed, there's no cancelation even though we end
    ##       up using these expressions
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
          # if the unit was on in t0, then it's m.PowerGeneratedT0[g] >= m.MinimumPowerOutput[g], and m.UnitOnT0 == 1
          # if not, then m.UnitOnT0[g] == 0 and so (m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]) * m.UnitOnT0[g] is 0
            return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] + \
                                                  (m.ScaledNominalRampUpLimit[g]+m.MinimumPowerOutput[g])*m.UnitStayOn[g,t] + \
    					      m.ScaledStartupRampLimit[g]*m.UnitStart[g,t] 
        else:
            return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedAboveMinimum[g, t-1] + \
                                                  (m.ScaledNominalRampUpLimit[g]+m.MinimumPowerOutput[g])*m.UnitStayOn[g,t] + \
    					        m.ScaledStartupRampLimit[g]*m.UnitStart[g,t] 
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    
    def enforce_ramp_down_limits_rule(m, g, t):
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                return m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g]*m.UnitStayOn[g,t] + (m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g])*m.UnitStop[g,t]
        else:
            return m.PowerGeneratedAboveMinimum[g, t-1] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g]*m.UnitStayOn[g,t] + (m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g])*m.UnitStop[g,t]
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

    return

@add_model_attr(component_name, requires = {'data_loader': None,
                                            'power_vars': None,
                                            'reserve_vars': None, 
                                            'generation_limits':generation_limits_w_startup_shutdown,
                                            })
def MLR_ramping(model):
    '''
    Equations (12) and (13) from 

    G. Morales-Espana, J. M. Latorre, and A. Ramos. Tight and compact MILP
    formulation for the thermal unit commitment problem. IEEE Transactions on
    Power Systems, 28(4):4897–4908, 2013.

    with T0 ramp-down limit which is required to make this consistent with other
    formulataions for ramping.

    '''

    # the following constraint encodes Constraint 12 defined in ME 
    
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
           return Constraint.Skip
        if t == m.InitialTime:
          # if the unit was on in t0, then it's m.PowerGeneratedT0[g] >= m.MinimumPowerOutput[g], and m.UnitOnT0 == 1
          # if not, then m.UnitOnT0[g] == 0 and so (m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]) * m.UnitOnT0[g] is 0
            return m.MaximumPowerAvailableAboveMinimum[g, t] <= m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] + \
                                                  m.ScaledNominalRampUpLimit[g] 
        else:
            return m.MaximumPowerAvailableAboveMinimum[g, t] <= m.PowerGeneratedAboveMinimum[g, t-1] + m.ScaledNominalRampUpLimit[g]
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    
    # the following constraint encodes Constraint 13 defined in ME
    
    def enforce_ramp_down_limits_rule(m, g, t):
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                return m.PowerGeneratedT0[g] - m.MinimumPowerOutput[g]*m.UnitOnT0[g] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g] 
        else:
            return m.PowerGeneratedAboveMinimum[g, t-1] - m.PowerGeneratedAboveMinimum[g, t] <= \
                        m.ScaledNominalRampDownLimit[g]
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

    ## need this so we agree with the other ramping models when using MLR Ramping
    ## (i.e., can't shutdown at t=1 unless we're below ScaledShutdownRampLimit)
    def power_limit_t0_stop_rule(m,g):
        if not m.enforce_t1_ramp_rates:
            return Constraint.Skip
        else:
            return m.PowerGeneratedT0[g] <= (m.MaximumPowerOutput[g])*m.UnitOnT0[g] \
                                            - (m.MaximumPowerOutput[g] - m.ScaledShutdownRampLimit[g])*m.UnitStop[g,m.InitialTime]
    model.power_limit_t0_stop = Constraint(model.ThermalGenerators,rule=power_limit_t0_stop_rule)

    return


## TODO: these really first appeared in Arroyo Conejo paper, and should be renamed thusly
@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': ['garver_3bin_vars','garver_2bin_vars','garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars'],
                                            'power_vars': None,
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def arroyo_conejo_ramping(model):

    '''
    equations (17) and (18) from

    J.M. Arroyo and A.J. Conejo, Optimal Response of a Thermal Unit 
    to an Electricity Spot Market, IEEE Transactions on Power Systems
    Vol. 15, No. 3, Aug 2000
    '''

    # impose upper bounds on the maximum power available for each generator in each time period,
    # based on standard and start-up ramp limits.
    
    # the following constraint encodes Constraint 6 defined in OAV
    
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from m.PowerGenerated[g, t-1])
        if t == m.InitialTime:
            return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] + \
                                                   m.ScaledNominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                                   m.ScaledStartupRampLimit[g] * m.UnitStart[g, t]
        else:
            return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g, t-1] + \
                                                  m.ScaledNominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                                  m.ScaledStartupRampLimit[g] * m.UnitStart[g,t]
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)

    # the following constraint encodes Constraint 7 defined in OAV
    
    def enforce_ramp_down_limits_rule(m, g, t):
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                     m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] + \
                     m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t]
        else:
            return m.PowerGenerated[g, t-1] - m.PowerGenerated[g, t] <= \
                 m.ScaledNominalRampDownLimit[g]  * m.UnitOn[g, t] + \
                 m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t]
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

def _OAV_enhanced(model):
    '''
    baseline for the OAV enhanced formulations
    '''

    
    # the following constraint encodes Constraint 23 defined in OAV
    
    def enforce_ramp_up_limits_rule(m, g, t):
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if (value(m.ScaledNominalRampUpLimit[g]) > value(m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g])) \
                and (value(m.ScaledMinimumUpTime[g]) >= 2):
            if t == m.InitialTime:
                return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] \
                                                       + m.ScaledNominalRampUpLimit[g] * m.UnitOn[g,t] \
                                                       + (m.ScaledStartupRampLimit[g]-m.ScaledNominalRampUpLimit[g]) * m.UnitStart[g, t] \
                                                       - m.MinimumPowerOutput[g]*m.UnitStop[g,t] \
                                                       - (m.ScaledNominalRampUpLimit[g]- m.ScaledShutdownRampLimit[g] + m.MinimumPowerOutput[g])*m.UnitStop[g,t+1]
            if t >= value(m.NumTimePeriods):
                return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g,t-1] \
                                                       + m.ScaledNominalRampUpLimit[g] * m.UnitOn[g,t] \
                                                       + (m.ScaledStartupRampLimit[g]-m.ScaledNominalRampUpLimit[g]) * m.UnitStart[g, t] \
                                                       - m.MinimumPowerOutput[g]*m.UnitStop[g,t]
            else:
                return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g,t-1] \
                                                       + m.ScaledNominalRampUpLimit[g] * m.UnitOn[g,t] \
                                                       + (m.ScaledStartupRampLimit[g]-m.ScaledNominalRampUpLimit[g]) * m.UnitStart[g, t] \
                                                       - m.MinimumPowerOutput[g]*m.UnitStop[g,t] \
                                                       - (m.ScaledNominalRampUpLimit[g]- m.ScaledShutdownRampLimit[g] + m.MinimumPowerOutput[g])*m.UnitStop[g,t+1]

        else: 
            if t == m.InitialTime:
               return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] + \
                                                       m.ScaledNominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                                       m.ScaledStartupRampLimit[g] * m.UnitStart[g, t]
            else:
               return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g, t-1] + \
                                                       m.ScaledNominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                                       m.ScaledStartupRampLimit[g] * m.UnitStart[g,t]
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_up_limits_rule)

    # the following constraint encodes Constraint 7, 20, 21 defined in OAV
    
    def enforce_ramp_down_limits_rule(m, g, t):
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                ## equation 7
                if (value(m.ScaledNominalRampDownLimit[g]) <= value(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g])) \
                        or (value(m.ScaledMinimumUpTime[g]) < 2):
                    return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                        m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] + \
                        m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t]
                elif value(m.ScaledMinimumUpTime[g]) < 3 or value(m.ScaledMinimumDownTime[g]) < 2: # now we can use equation 20
                    return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] \
                        + m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t] \
                        - (m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g])*m.UnitStart[g,t]
                else: # we can use equation (21)
                    return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t+1] \
                        + m.ScaledShutdownRampLimit[g] * m.UnitStop[g, t] \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitStop[g,t+1] \
                        -(m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g]) * m.UnitStart[g,t] \
                        - m.ScaledNominalRampDownLimit[g] * m.UnitStart[g,t+1]

        else:
                ## equation 7
                if (value(m.ScaledNominalRampDownLimit[g]) <= value(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g])) \
                        or (value(m.ScaledMinimumUpTime[g]) < 2):
                    return m.PowerGenerated[g,t-1] - m.PowerGenerated[g, t] <= \
                        m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] + \
                        m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t]
                elif value(m.ScaledMinimumUpTime[g]) < 3 or value(m.ScaledMinimumDownTime[g]) < 2 or t >= value(m.NumTimePeriods): # now we can use equation 20
                    return m.PowerGenerated[g,t-1] - m.PowerGenerated[g, t] <= \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] \
                        + m.ScaledShutdownRampLimit[g]  * m.UnitStop[g, t] \
                        -(m.ScaledNominalRampDownLimit[g]-m.ScaledStartupRampLimit[g]+m.MinimumPowerOutput[g])*m.UnitStart[g,t-1] \
                        - (m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g])*m.UnitStart[g,t]
                else: # we can use equation (21)
                    return m.PowerGenerated[g,t-1] - m.PowerGenerated[g, t] <= \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t+1] \
                        + m.ScaledShutdownRampLimit[g] * m.UnitStop[g, t] \
                        + m.ScaledNominalRampDownLimit[g] * m.UnitStop[g,t+1] \
                        -(m.ScaledNominalRampDownLimit[g]-m.ScaledStartupRampLimit[g]+m.MinimumPowerOutput[g])*m.UnitStart[g,t-1] \
                        -(m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g]) * m.UnitStart[g,t] \
                        - m.ScaledNominalRampDownLimit[g] * m.UnitStart[g,t+1]
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)

## TODO: These should really be refactored so we don't double- or triple-up on ramping limits
@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': None,
                                            'status_vars': ['garver_3bin_vars','garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars'],
                                            'power_vars': None,
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def OAV_ramping_enhanced(model):

    '''
    Equations (6),(7),(20),(21),(23) from 

    Ostrowski, J., et. al. Tight Mixed Integer Linear Programming Formulations
    for the Unit Commitment Problem. IEEE Transactions on Power Systems, 
    Vol. 27, No. 1, Feb 2012.

    We only add the strongest valid ramp-up or ramp-down equality we can,
    and discard the others
    '''

    _OAV_enhanced(model)


@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': ['garver_3bin_vars','garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars'],
                                            'power_vars': None,
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def OAV_ramping_enhanced_2period(model):

    '''
    This is OAV_ramping_enhanced plus the two-period ramping inequalities
    in equations (22) and (24) from

    Ostrowski, J., et. al. Tight Mixed Integer Linear Programming Formulations
    for the Unit Commitment Problem. IEEE Transactions on Power Systems, 
    Vol. 27, No. 1, Feb 2012.

    '''

    #TODO: This isn't quite valid, needs debugging
    #NOTE: I think I had fixed this?? Need to check
    _OAV_enhanced(model)

    ## TODO: this shouldn't be necessary, and the MaximumPowerAvailable
    ##       should be on the LHS of these equations
    def OAV_two_period_ramp_up_rule(m,g,t):
        if 2*value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
            return Constraint.Skip
        if (value(m.ScaledNominalRampUpLimit[g]) > value(m.ScaledShutdownRampLimit[g] - m.MinimumPowerOutput[g])) \
                and (value(m.ScaledMinimumUpTime[g]) >= 2) and (value(m.ScaledMinimumDownTime[g]) >= 2) \
                and (t > value(m.InitialTime)):
            if t == value(m.InitialTime) + 1: ## t == 2
                return m.MaximumPowerAvailable[g,t] - m.PowerGeneratedT0[g] <= \
                          2 * m.ScaledNominalRampUpLimit[g] * m.UnitOn[g,t] \
                        - m.MinimumPowerOutput[g]*(m.UnitStop[g,t-1]+m.UnitStop[g,t]) \
                        + (m.ScaledStartupRampLimit[g] - m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t-1] \
                        + (m.ScaledStartupRampLimit[g] - 2*m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t]
            else:
                return m.MaximumPowerAvailable[g,t] - m.PowerGenerated[g,t-2] <= \
                          2 * m.ScaledNominalRampUpLimit[g] * m.UnitOn[g,t] \
                        - m.MinimumPowerOutput[g]*(m.UnitStop[g,t-1]+m.UnitStop[g,t]) \
                        + (m.ScaledStartupRampLimit[g] - m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t-1] \
                        + (m.ScaledStartupRampLimit[g] - 2*m.ScaledNominalRampUpLimit[g])*m.UnitStart[g,t]
        else:
            return Constraint.Skip
    model.OAVTwoPeriodRampUp = Constraint(model.ThermalGenerators, model.TimePeriods, rule=OAV_two_period_ramp_up_rule)

    ## NOTE: in the text this doesn't have any conditions on when it is valid,
    ##       so the valid conditions were inferred
    def OAV_two_period_ramp_down_rule(m,g,t):
        if 2*value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]):
            return Constraint.Skip
        if (value(m.ScaledNominalRampDownLimit[g]) > value(m.ScaledStartupRampLimit[g] - m.MinimumPowerOutput[g])) \
                and (value(m.ScaledMinimumUpTime[g]) >= 3) and (value(m.ScaledMinimumDownTime[g]) >= 2) \
                and (t > value(m.InitialTime)):
            if t == value(m.InitialTime) + 1: ## t == 2
                if not m.enforce_t1_ramp_rates:
                    return Constraint.Skip
                return m.PowerGeneratedT0[g] - m.PowerGenerated[g,t] <= \
                          2*m.ScaledNominalRampDownLimit[g]*m.UnitOn[g,t] \
                        + m.ScaledShutdownRampLimit[g]*m.UnitStop[g,t-1] \
                        + (m.ScaledShutdownRampLimit[g]+m.ScaledNominalRampDownLimit[g])*m.UnitStop[g,t] \
                        -(2*m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g])*(m.UnitStart[g,t-1]+m.UnitStart[g,t])
            else:
                return m.PowerGenerated[g,t-2] - m.PowerGenerated[g,t] <= \
                          2*m.ScaledNominalRampDownLimit[g]*m.UnitOn[g,t] \
                        + m.ScaledShutdownRampLimit[g]*m.UnitStop[g,t-1] \
                        + (m.ScaledShutdownRampLimit[g]+m.ScaledNominalRampDownLimit[g])*m.UnitStop[g,t] \
                        - 2*m.ScaledNominalRampDownLimit[g]*m.UnitStart[g,t-2] \
                        -(2*m.ScaledNominalRampDownLimit[g]+m.MinimumPowerOutput[g])*(m.UnitStart[g,t-1]+m.UnitStart[g,t])
        else:
            return Constraint.Skip
                
    model.OAVTwoPeriodRampDown = Constraint(model.ThermalGenerators, model.TimePeriods, rule=OAV_two_period_ramp_down_rule)


@add_model_attr(component_name, requires = {'data_loader': None,
                                            'status_vars': None,
                                            'power_vars': None,
                                            'reserve_vars': None,
                                            'generation_limits':None,
                                            })
def CA_ramping_limits(model):

    '''
    Equations (18),(19) and (20) from

    Carrion, M. and Arroyo, J. (2006) A Computationally Efficient Mixed-Integer
    Liner Formulation for the Thermal Unit Commitment Problem. IEEE Transactions
    on Power Systems, Vol. 21, No. 3, Aug 2006.
    '''

    # impose upper bounds on the maximum power available for each generator in each time period,
    # based on standard and start-up ramp limits.
    
    # the following constraint encodes Constraint 18 defined in Carrion and Arroyo.
    
    def enforce_max_available_ramp_up_rates_rule(m, g, t):
       # 4 cases, split by (t-1, t) unit status (RHS is defined as the delta from m.PowerGenerated[g, t-1])
       # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound due to unit being off)
       # (0, 1) - unit switching on:  RHS = startup ramp limit
       # (1, 0) - unit switching off: RHS = standard ramp limit minus startup ramp limit plus maximum power output (degenerate upper bound due to unit off)
       # (1, 1) - unit staying on:    RHS = standard ramp limit plus power generated in previous time period
        if value(m.ScaledNominalRampUpLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            return m.MaximumPowerAvailable[g, t] <= m.PowerGeneratedT0[g] + \
                                                   m.ScaledNominalRampUpLimit[g] * m.UnitOnT0[g] + \
                                                   m.ScaledStartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOnT0[g]) + \
                                                   m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
        else:
            return m.MaximumPowerAvailable[g, t] <= m.PowerGenerated[g, t-1] + \
                                                  m.ScaledNominalRampUpLimit[g] * m.UnitOn[g, t-1] + \
                                                  m.ScaledStartupRampLimit[g] * (m.UnitOn[g, t] - m.UnitOn[g, t-1]) + \
                                                  m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t])
    
    model.EnforceMaxAvailableRampUpRates = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_max_available_ramp_up_rates_rule)
    
    
    # the following constraint encodes Constraint 20 defined in Carrion and Arroyo.
    
    def enforce_ramp_down_limits_rule(m, g, t):
        # 4 cases, split by (t-1, t) unit status:
        # (0, 0) - unit staying off:   RHS = maximum generator output (degenerate upper bound)
        # (0, 1) - unit switching on:  RHS = standard ramp-down limit minus shutdown ramp limit plus maximum generator output - this is the strangest case.
        #NOTE: This may never be physically true, but if a generator has ScaledShutdownRampLimit >> MaximumPowerOutput, this constraint causes problems
        # (1, 0) - unit switching off: RHS = shutdown ramp limit
        # (1, 1) - unit staying on:    RHS = standard ramp-down limit
        if value(m.ScaledNominalRampDownLimit[g]) >= value(m.MaximumPowerOutput[g] - m.MinimumPowerOutput[g]) and model.generation_limits in generation_limits_w_startup_shutdown:
            return Constraint.Skip
        if t == m.InitialTime:
            if not m.enforce_t1_ramp_rates:
                return Constraint.Skip
            else:
                return m.PowerGeneratedT0[g] - m.PowerGenerated[g, t] <= \
                     m.ScaledNominalRampDownLimit[g] * m.UnitOn[g, t] + \
                     m.ScaledShutdownRampLimit[g]  * (m.UnitOnT0[g] - m.UnitOn[g, t]) + \
                     m.MaximumPowerOutput[g] * (1 - m.UnitOnT0[g])
        else:
            return m.PowerGenerated[g, t-1] - m.PowerGenerated[g, t] <= \
                 m.ScaledNominalRampDownLimit[g]  * m.UnitOn[g, t] + \
                 m.ScaledShutdownRampLimit[g]  * (m.UnitOn[g, t-1] - m.UnitOn[g, t]) + \
                 m.MaximumPowerOutput[g] * (1 - m.UnitOn[g, t-1])
    
    model.EnforceScaledNominalRampDownLimits = Constraint(model.ThermalGenerators, model.TimePeriods, rule=enforce_ramp_down_limits_rule)
