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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as clrs
import matplotlib.cm as cmap
import seaborn as sns
from cycler import cycler
import math
import unittest
import logging
import egret.data.test_utils as tu
import egret.models.tests.test_approximations as ta
from pyomo.opt import SolverFactory, TerminationCondition
from egret.models.lccm import *
from egret.models.dcopf_losses import *
from egret.models.dcopf import *
from egret.models.copperplate_dispatch import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join


def read_sensitivity_data(case_folder, test_model, data_generator=tu.total_cost):
    parent, case = os.path.split(case_folder)
    filename = case + "_" + test_model + "_*.json"
    file_list = glob.glob(os.path.join(case_folder, filename))

    data_type = data_generator.__name__

    print("Reading " + data_type + " data from " + filename + ".")

    data = {}
    for file in file_list:
        md_dict = json.load(open(file))
        md = ModelData(md_dict)
        mult = md.data['system']['mult']
        data[mult] = data_generator(md)

    data_is_vector = False
    for d in data:
        data_is_vector = hasattr(data[d], "__len__")

    if data_is_vector:
        df_data = pd.DataFrame(data)
        df_data = df_data.sort_index(axis=1)
        # print('data: {}'.format(df_data))
    else:
        df_data = pd.DataFrame(data, index=[test_model])
        # df_data = df_data.transpose()
        df_data = df_data.sort_index(axis=1)
        # print('data: {}'.format(df_data))

    return df_data


def geometricMean(array):

    geomean = list()

    for row in array:
        n = len(row)
        sum = 0
        for i in range(n):
            sum += math.log(row[i])
        sum = sum / n
        gm = math.exp(sum)
        geomean.append(gm)

    return geomean


def generate_mean_data(test_case, test_model_dict, function_list=[tu.num_buses,tu.num_constraints,tu.model_density,tu.acpf_slack,tu.solve_time]):

    case_location = ta.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    ## include acopf results
    test_model_dict['acopf'] = True

    df_data = pd.DataFrame(data=None, index=test_model_dict.keys())

    for func in function_list:
        ## put data into blank DataFrame
        df_func = pd.DataFrame(data=None)

        for test_model, val in test_model_dict.items():
            # read data and place in df_func
            df_raw = read_sensitivity_data(case_location, test_model, data_generator=func)
            df_func = pd.concat([df_func , df_raw], sort=True)

        func_name = func.__name__

        ## also calculate geomean and maximum if function is solve_time()
        if func_name=='solve_time':
            gm = geometricMean(df_func.to_numpy())
            gm_name = func_name + '_geomean'
            df_gm = pd.DataFrame(data=gm, index=df_func.index, columns=[gm_name])
            df_data[gm_name] = df_gm

            max = df_func.max(axis=1)
            max_name = func_name + '_max'
            df_max = pd.DataFrame(data=max.values, index=df_func.index, columns=['max_' + func_name])
            df_data[max_name] = df_max

            func_name = func_name + '_avg'


        df_func = df_func.mean(axis=1)
        df_func = pd.DataFrame(data=df_func.values, index=df_func.index, columns=[func_name])

        #df_data = pd.concat([df_data, df_func])
        df_data[func_name] = df_func

    ## save DATA to csv
    destination = ta.get_summary_file_location('data')
    filename = "mean_data_" + case_name + ".csv"
    df_data.to_csv(os.path.join(destination, filename))


def generate_speedup_data(test_model_dict, case_list=None, mean_data='solve_time_geomean', benchmark='dlopf_lazy'):

    ## get data
    if case_list is None:
        case_list = ta.get_case_names()

    data_dict = {}
    cases = []
    for case in case_list:
        try:
            input = "mean_data_" + case + ".csv"
            df_data = get_data(input, test_model_dict)
            models = list(df_data.index.values)
            for m in models:
                val = df_data.at[benchmark, mean_data] / df_data.at[m, mean_data]
                if m in data_dict:
                    data_dict[m].append(val)
                else:
                    data_dict[m] = [val]
            cases.append(case)
        except:
            pass

    df_data = pd.DataFrame(data_dict,index=cases)
    df_data.loc['AVERAGE'] = df_data.mean()

    ## save DATA to csv
    destination = ta.get_summary_file_location('data')
    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data.to_csv(os.path.join(destination, filename))


def generate_speedup_heatmap(test_model_dict, mean_data='solve_time_geomean', benchmark='dlopf_lazy',colormap=None,
                             cscale='linear', include_benchmark=False, show_plot=False):

    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data = get_data(filename,test_model_dict=test_model_dict)
    if not include_benchmark:
        df_data = df_data.drop(columns=benchmark)

    cols = df_data.columns.to_list()
    col_lazy=[]
    col_alert=[]
    for c in cols:
        if 'lazy' in c:
            col_lazy.append(c)
        else:
            col_alert.append(c)
    cols = col_alert + col_lazy
    df_data = df_data[cols]

    model_names = [c for c in df_data.columns]
#    index_names = [i for i in df_data.index]
    index_names = [i.replace('pglib_opf_','') for i in df_data.index]
    data = df_data.values
    model_num = len(model_names)

    #   EDIT TICKS HERE IF NEEDED   #
    if cscale == 'log':
        cbar_dict = {'ticks' : [1e0,1e1,1e2]}
        cbar_norm = clrs.LogNorm(vmin=data.min(), vmax=data.max())
    else:
        cbar_dict = None
        cbar_norm = None

    ax = sns.heatmap(data,
                     linewidth=0.,
                     xticklabels=model_names,
                     yticklabels=index_names,
                     cmap=colormap,
                     norm=cbar_norm,
                     cbar_kws=cbar_dict,
                     )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_title(mean_data + " speedup vs. " + benchmark)
    ax.set_xlabel("Model")
    ax.set_ylabel("Test Case")

    plt.tight_layout()

    ## save FIGURE as png
    filename = "speedupplot_v_" + benchmark + "_" + mean_data + ".png"
    destination = ta.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()
        plt.clf()


def generate_sensitivity_data(test_case, test_model_dict, data_generator=tu.acpf_slack,
                              data_is_pct=False, data_is_vector=False, vector_norm=2):

    case_location = ta.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    # acopf comparison
    df_acopf = read_sensitivity_data(case_location, 'acopf', data_generator=data_generator)


    ## calculates specified L-norm of difference with acopf (e.g., generator dispatch, branch flows, voltage profile...)
    if data_is_vector:
        print('data is vector of length {}'.format(len(df_acopf.values)))

    ## calcuates relative difference from acopf (e.g., objective value, solution time...)
    elif data_is_pct:
        acopf_avg = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
        print('data is pct with acopf values averaging {}'.format(acopf_avg))
    else:
        acopf_avg = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
        print('data is nominal with acopf values averaging {}'.format(acopf_avg))

    # empty dataframe to add data into
    df_data = pd.DataFrame(data=None)

    # iterate over test_model's
    test_model_dict['acopf'] = True
    for test_model, val in test_model_dict.items():
        if val:
            df_approx = read_sensitivity_data(case_location, test_model, data_generator=data_generator)

            # calculate norm from df_diff columns
            data = {}
            avg_ac_data = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
            for col in df_approx:
                if data_is_vector is True:
                    data[col] = np.linalg.norm(df_approx[col].values - df_acopf[col].values, vector_norm)
                elif data_is_pct is True:
                    data[col] = ((df_approx[col].values - df_acopf[col].values) / df_acopf[col].values) * 100
                else:
                    data[col] = df_approx[col].values

            # record test_model column in DataFrame
            df_col = pd.DataFrame(data, index=[test_model])
            df_data = pd.concat([df_data, df_col], sort=True)


    ## save DATA as csv
    y_axis_data = data_generator.__name__
    df_data = df_data.T
    destination = ta.get_summary_file_location('data')
    filename = "sensitivity_data_" + case_name + "_" + y_axis_data + ".csv"
    df_data.to_csv(os.path.join(destination, filename))



def get_data(filename, test_model_dict):
    print(filename)

    ## get data from CSV
    source = ta.get_summary_file_location('data')
    df_data = pd.read_csv(os.path.join(source,filename), index_col=0)

    remove_list = []
    for tm,val in test_model_dict.items():
        if not val:
            remove_list.append(tm)

    for rm in remove_list:
        if rm in df_data.index:
            df_data = df_data.drop(rm, axis=0)
        elif rm in df_data.columns:
            df_data = df_data.drop(rm, axis=1)

    return df_data

def generate_pareto_plot(test_case, test_model_dict, y_data='acpf_slack', x_data='solve_time', y_units='p.u', x_units='s',
                         mark_default='o', mark_lazy='D', mark_acopf='*', mark_size=36, colors=cmap.viridis,
                         annotate_plot=False, show_plot=False):

    ## get data
    _, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)
    input = "mean_data_" + case_name + ".csv"
    df_data = get_data(input, test_model_dict)

    models = list(df_data.index.values)
    df_y_data = df_data[y_data]
    df_x_data = df_data[x_data]


    ## assign color values
    #num_entries = len(df_data)
    #color = colors(np.linspace(0, 1, num_entries))
    #custom_cycler = (cycler(color=color))
    #plt.rc('axes', prop_cycle=custom_cycler)

    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    markers = ['o', '+', 'x']
    if colors is not None:
        n = len(df_data)
        m = len(markers)
        ax.set_prop_cycle(color=[colors(i) for i in np.linspace(0,1,n)],
                          marker=[markers[i%m] for i in range(n)])
    else:
        ax.set_prop_cycle(marker=markers)

    for m in models:
        if 'lazy' in m:
            mark = mark_lazy
        elif 'acopf' in m:
            mark = mark_acopf
        else:
            mark = mark_default

        x = df_x_data[m]
        y = df_y_data[m]
        ax.scatter(x, y, s=mark_size, label=m, marker=mark)

        if annotate_plot:
            ax.annotate(m, (x,y))

    ax.set_title(y_data + " vs. " + x_data + "\n(" + case_name + ")")
    ax.set_ylabel(y_data + " (" + y_units + ")")
    ax.set_xlabel(x_data + " (" + x_units + ")")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ## save FIGURE to png
    figure_dest = ta.get_summary_file_location('figures')
    filename = "paretoplot_" + case_name + "_" + y_data + "_v_" + x_data + ".png"
    plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()



def generate_sensitivity_plot(test_case, test_model_dict, plot_data='acpf_slack', units='p.u.',
                              colors=cmap.viridis, show_plot=False):

    ## get data
    _, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)
    input = "sensitivity_data_" + case_name + "_" + plot_data + ".csv"
    df_data = get_data(input, test_model_dict)

    models = list(df_data.columns.values)


    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    markers = ['x','o','+']
    if colors is not None:
        n = len(df_data.columns)
        m = len(markers)
        ax.set_prop_cycle(color=[colors(i) for i in np.linspace(0,1,n)],
                          marker=[markers[i%m] for i in range(n)])
    else:
        ax.set_prop_cycle(marker=markers)

    for m in models:
        y = df_data[m]
        if m =='acopf':
            ax.plot(y, label=m, marker='')
        else:
            ax.plot(y, label=m)


    ax.set_title(plot_data + " (" + case_name + ")")
    # output.set_ylim(top=0)
    ax.set_xlabel("Demand Multiplier")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel(plot_data + " (" +  units + ")")

    ## save FIGURE as png
    destination = ta.get_summary_file_location('figures')
    filename = "sensitivityplot_" + case_name + "_" + plot_data + ".png"
    plt.savefig(os.path.join(destination, filename))

    # display
    if show_plot is True:
        plt.show()
    else:
        plt.cla()


def generate_case_size_plot_seaborn(test_model_dict, case_list=None,
                            y_data='solve_time_geomean', y_units='s',
                            x_data='num_buses', x_units=None,
                            s_data=None, s_units=None,
                            colors=cmap.viridis, s_max=250, s_min=1, x_min = 0,
                            yscale='linear',xscale='linear',
                            annotate_plot=False, show_plot=False):

    ## get data
    if case_list is None:
        case_list = ta.get_case_names()
    if s_data is None:
        var_names = ['model',x_data,y_data]
    else:
        var_names = ['model',x_data,y_data,s_data]
    sns_data = pd.DataFrame(data=None,columns=var_names)
    cases = []
    for case in case_list:
        try:
            input = "mean_data_" + case + ".csv"
            df_data = get_data(input, test_model_dict)
            if 'con_per_bus' in var_names:
                df_data['con_per_bus'] = df_data['num_constraints'] / df_data['num_buses']

            models = list(df_data.index.values)
            df_data['model'] = models

            var_drop = [var for var in df_data.columns if var not in var_names]
            df_data = df_data.drop(labels=var_drop, axis=1)

            sns_data = sns_data.append(df_data, ignore_index=True)
            cases.append(case)

        except:
            pass

    sns.set(style="ticks", palette='colorblind')
    if s_data is None:
        g = sns.scatterplot(x=x_data, y=y_data, size=s_data, hue='model', style='model', data=sns_data)
    else:
        g = sns.scatterplot(x=x_data, y=y_data, size=s_data, hue='model', data=sns_data)
    sns.despine()

    plt.yscale(yscale)
    plt.xscale(xscale)
    #plt.tight_layout()

    # set legend location
    box = g.get_position()
    g.set_position([box.x0, box.y0, 0.75 * box.width, box.height])
    plt.legend(bbox_to_anchor=(1, 0.5), loc=6)


    ## save FIGURE to png
    figure_dest = ta.get_summary_file_location('figures')
    filename = "casesizeplot_" + y_data + "_v_" + x_data + ".png"
    plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()


def generate_case_size_plot(test_model_dict, case_list=None,
                            y_data='solve_time_geomean', y_units='s',
                            x_data='num_buses', x_units=None,
                            s_data=None, colors=cmap.viridis, s_max=250, s_min=1,
                            yscale='linear',xscale='linear',
                            annotate_plot=False, show_plot=False):

    ## get data
    if case_list is None:
        case_list = ta.get_case_names()
    y_dict = {}
    x_dict = {}
    s_dict = {}
    if s_data is None:
        var_names = ['model',x_data,y_data]
    else:
        var_names = ['model',x_data,y_data,s_data]
    for case in case_list:
        try:
            input = "mean_data_" + case + ".csv"
            df_data = get_data(input, test_model_dict)
            if 'con_per_bus' in var_names:
                df_data['con_per_bus'] = df_data['num_constraints'] / df_data['num_buses']
            models = list(df_data.index.values)
            for m in models:
                if m in y_dict.keys():
                    y_dict[m].append(df_data.at[m, y_data])
                    x_dict[m].append(df_data.at[m, x_data])
                    if s_data is None:
                        s_dict[m].append(36)
                    else:
                        s_dict[m].append(df_data.at[m, s_data])
                else:
                    y_dict[m] = [df_data.at[m, y_data]]
                    x_dict[m] = [df_data.at[m, x_data]]
                    if s_data is None:
                        s_dict[m] = [36]
                    else:
                        s_dict[m] = [df_data.at[m, s_data]]

        except:
            pass

    df_y_data = pd.DataFrame(y_dict).fillna(0)
    df_x_data = pd.DataFrame(x_dict).fillna(0)
    df_s_data = pd.DataFrame(s_dict).fillna(0)

    # scale s_data
    arr = df_s_data.values
    data_max = arr.max()
    arr = arr * (s_max / data_max)
    arr[arr<s_min] = s_min
    df_s_data = pd.DataFrame(data=arr, columns=models)


    ## Create plot
    fig, ax = plt.subplots(figsize=(9, 4))

    #---- set color cycle
    if colors is not None:
        ax.set_prop_cycle(color=[colors(i) for i in np.linspace(0,1,len(df_data))])

    #---- plot data
    for m in models:

        x = df_x_data[m]
        y = df_y_data[m]
        mark_size = df_s_data[m]
        ax.scatter(x, y, s=mark_size, label=None)

        if annotate_plot:
            ax.annotate(m, (x,y))

    # ---- plot empty data to help format the legend
    for m in models:
        x = []
        y = []
        mark_size = None
        ax.scatter(x, y, s=mark_size, label=m)


    #ax.set_title(y_data + " vs. " + x_data)
    if y_units is None:
        ax.set_ylabel(y_data)
    else:
        ax.set_ylabel(y_data + " (" + y_units + ")")
    if x_units is None:
        ax.set_xlabel(x_data)
    else:
        ax.set_xlabel(x_data + " (" + x_units + ")")

    plt.yscale(yscale)
    plt.xscale(xscale)
    x1,x2 = ax.get_xlim()
    ax.set_xlim([2,x2])
    plt.tight_layout()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    first_legend = plt.legend(title='models', bbox_to_anchor=(1, 0.35), loc='lower left')
    plt.gca().add_artist(first_legend)

    if s_data is not None:
        lgd_title = s_data
        create_circlesize_legend(title=lgd_title,s_min=s_min, s_max=s_max, data_max=data_max)


    ## save FIGURE to png
    figure_dest = ta.get_summary_file_location('figures')
    filename = "casesizeplot_" + y_data + "_v_" + x_data + ".png"
    plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()
        plt.clf()
        plt.close(fig)


def create_circlesize_legend(title=None, s_min=1, s_max=500, data_min=2, data_max=1000):

    c = '0.75'
    sizes = np.linspace(s_min, s_max, num=4)
    data = np.linspace(data_min, data_max, num=4)
#    if s_min <= 0:
#        s_min=0.1
#    if data_min <= 0:
#        data_min = 0.1
#    sizes = np.logspace(np.log10(s_min), np.log10(s_max), num=4)
#    data = np.logspace(np.log10(data_min), np.log10(data_max), num=4)

    dots = [plt.scatter([], [], color=c, s=sizes[i]) for i in range(len(sizes))]
    labels = [str(int(round(data[i],0))) for i in range(len(sizes))]

    new_legend = plt.legend(dots, labels,title=title, bbox_to_anchor=(1.05,0.35), loc='upper left')
    plt.gca().add_artist(new_legend)


def create_full_summary(test_case, test_model_dict, show_plot=True):
    """
    solves generates plots for test_case
    """

    ## Sequential colors: lightness value increases monotonically
    #colors = cmap.viridis #*****#
    #colors = cmap.cividis
    #colors = cmap.magma
    #colors = cmap.plasma #*****#
    ## Diverging/cyclic colors: monotonically increasing lightness followed by monotonically decreasing lightness
    #colors = cmap.Spectral #*****#
    #colors = cmap.coolwarm #*****#
    #colors = cmap.twilight
    #colors = cmap.twilight_shifted
    #colors = cmap.hsv
    ## Qualitative colors: not perceptual
    #colors = cmap.Paired
    #colors = cmap.Accent
    #colors = cmap.Set3
    ## Miscellaneous colors:
    colors = cmap.gnuplot #*****#
    #colors = cmap.jet
    #colors = cmap.nipy_spectral #*****#

    # remove unused models from reporting
    test_model_dict = {key:val for key,val in test_model_dict.items() if val}

    mean_functions = [tu.num_buses,
                      tu.num_branches,
                      tu.num_constraints,
                      tu.num_variables,
                      tu.model_density,
                      tu.solve_time,
                      tu.acpf_slack,
                      tu.avg_vm_UB_viol,
                      tu.avg_vm_LB_viol,
                      tu.avg_vm_viol,
                      tu.avg_thermal_viol,
                      tu.max_vm_UB_viol,
                      tu.max_vm_LB_viol,
                      tu.max_vm_viol,
                      tu.max_thermal_viol,
                      tu.pct_vm_UB_viol,
                      tu.pct_vm_LB_viol,
                      tu.pct_vm_viol,
                      tu.pct_thermal_viol,
                      ]

    sum_functions = [tu.optimal,
                     tu.infeasible,
                     tu.maxTimeLimit,
                     tu.maxIterations,
                     tu.solverFailure,
                     tu.internalSolverError,
    #                 tu.dualfailed,
                     ]

    ## Generate data files
    #generate_mean_data(test_case,test_model_dict) ## to just grab the default metrics
    generate_mean_data(test_case,test_model_dict, function_list=mean_functions)
    generate_sensitivity_data(test_case, test_model_dict, data_generator=tu.acpf_slack)

    ## Generate plots
    #---- Sensitivity plots: remove lazy and tolerance models
    for key, val in test_model_dict.items():
        if 'lazy' in key or '_e' in key:
            test_model_dict[key] = False
    generate_sensitivity_plot(test_case, test_model_dict, plot_data='acpf_slack', units='MW', colors=colors, show_plot=show_plot)

    #---- Pareto plots: add lazy models
    for key, val in test_model_dict.items():
        if 'lazy' in key:
            test_model_dict[key] = False
        elif 'default' in key:
            test_model_dict[key] = True
    generate_pareto_plot(test_case, test_model_dict, y_data='acpf_slack', x_data='solve_time_geomean', y_units='MW', x_units='s',
                         mark_default='o', mark_lazy='+', mark_acopf='*', mark_size=100, colors=colors,
                         annotate_plot=False, show_plot=show_plot)

    #---- Case size plots:
    generate_case_size_plot(test_model_dict,y_data='solve_time_geomean', y_units='s',
                            x_data='num_buses', x_units=None, s_data='con_per_bus',colors=colors,
                            xscale='log', yscale='linear',show_plot=show_plot)

    #---- Lazy model speedup: remove all but default and lazy models
    for key, val in test_model_dict.items():
        if 'acopf' in key \
                or 'slopf' in key \
                or 'dlopf_default' in key \
                or 'dlopf_lazy' in key \
                or 'clopf_default' in key \
                or 'clopf_lazy' in key \
                or 'clopf_p_default' in key \
                or 'clopf_p_lazy' in key \
                or 'dcopf_btheta' in key:
            test_model_dict[key] = True
        else:
            test_model_dict[key] = False
    generate_speedup_data(test_model_dict, mean_data='solve_time_geomean', benchmark='acopf')
    generate_speedup_heatmap(test_model_dict, mean_data='solve_time_geomean', benchmark='acopf',
                             colormap=None, cscale='log', show_plot=show_plot)

    # ---- Factor truncation speedup: remove all but default and tolerance option models
    for key, val in test_model_dict.items():
        if 'dlopf_default' in key \
                or 'dlopf_e' in key \
                or 'clopf_default' in key \
                or 'clopf_e' in key:
            test_model_dict[key] = True
        else:
            test_model_dict[key] = False
        if '_e2' in key:
            test_model_dict[key] = False
    generate_speedup_data(test_model_dict, mean_data='solve_time_geomean', benchmark='dlopf_default')
    generate_speedup_heatmap(test_model_dict, mean_data='solve_time_geomean', benchmark='dlopf_default',colormap=None,
                             cscale='linear', include_benchmark=True, show_plot=show_plot)

    #---- Model sparsity plot
    for key, val in test_model_dict.items():
        if 'slopf' in key \
                or 'dlopf_default' in key \
                or 'dlopf_e' in key \
                or 'clopf_default' in key \
                or 'clopf_e' in key \
                or 'clopf_p_default' in key \
                or 'clopf_p_e' in key \
                or 'dcopf_ptdf_default' in key \
                or 'dcopf_ptdf_e' in key \
                or 'dcopf_btheta' in key:
            test_model_dict[key] = True
        else:
            test_model_dict[key] = False
        if '_e2' in key:
            test_model_dict[key] = False
    generate_pareto_plot(test_case, test_model_dict, y_data='solve_time_geomean', x_data='model_density', y_units='s', x_units='%',
                         mark_default='o', mark_lazy='+', mark_acopf='*', mark_size=100, colors=colors,
                         annotate_plot=False, show_plot=show_plot)


if __name__ == '__main__':
    test_case = ta.idx_to_test_case(0)
    test_model_dict = \
        {'acopf' : True,
         'slopf': True,
         'dlopf_default': True,
         'dlopf_lazy' : True,
         'dlopf_e5': True,
         'dlopf_e4': True,
         'dlopf_e3': True,
         'dlopf_e2': False,
         'clopf_default': True,
         'clopf_lazy': True,
         'clopf_e5': True,
         'clopf_e4': True,
         'clopf_e3': True,
         'clopf_e2': False,
         'clopf_p_default': True,
         'clopf_p_lazy': True,
         'clopf_p_e5': True,
         'clopf_p_e4': True,
         'clopf_p_e3': True,
         'clopf_p_e2': False,
         'qcopf_btheta': True,
         'dcopf_ptdf_default': True,
         'dcopf_ptdf_lazy': True,
         'dcopf_ptdf_e5': True,
         'dcopf_ptdf_e4': True,
         'dcopf_ptdf_e3': True,
         'dcopf_ptdf_e2': False,
         'dcopf_btheta': True
         }

    create_full_summary(test_case,test_model_dict)