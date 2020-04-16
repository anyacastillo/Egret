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
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
import math
import unittest
import logging
import egret.data.test_utils as tu
import egret.models.tests.ta_utils as tau
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

# Functions to be summarized by averaging
mean_functions = [tu.num_buses,
                  tu.num_branches,
                  tu.num_constraints,
                  tu.num_variables,
                  tu.model_density,
                  tu.solve_time,
                  tu.acpf_slack,
                  tu.vm_UB_viol_avg,
                  tu.vm_LB_viol_avg,
                  tu.vm_viol_avg,
                  tu.thermal_viol_avg,
                  tu.vm_UB_viol_max,
                  tu.vm_LB_viol_max,
                  tu.vm_viol_max,
                  tu.thermal_viol_max,
                  tu.vm_UB_viol_pct,
                  tu.vm_LB_viol_pct,
                  tu.vm_viol_pct,
                  tu.thermal_viol_pct,
                  ]

#Functions to be summarized by summation
sum_functions = [tu.optimal,
                 tu.infeasible,
                 tu.maxTimeLimit,
                 tu.maxIterations,
                 tu.solverFailure,
                 tu.internalSolverError,
                 tu.duals,
                 ]

summary_functions = {}
sf = summary_functions
for func in mean_functions:
    key = func.__name__
    sf[key] = {'function' : func, 'summarizers' : ['avg']}
for func in sum_functions:
    key = func.__name__
    sf[key] = {'function' : func, 'summarizers' : ['sum']}
sf['solve_time']['summarizers'] = ['avg','geomean','max']
sf['acpf_slack']['summarizers'] = ['avg','max']

def get_colors(map_name=None, trim=0.9):

    if map_name is None:
        map_name = 'gnuplot'

    trim_top = [
        'ocean',
        'gist_earth',
        'terrain',
        'gnuplot2',
        'CMRmap',
        'cubehelix'
    ]

    colors = cmap.get_cmap(name=map_name)

    if map_name in trim_top:
        trim_colors = ListedColormap(colors(np.linspace(0,trim,256)))
        return trim_colors

    colors.set_bad('grey')

    return colors

def short_summary():

    function_list = ['num_buses', 'num_constraints', 'acpf_slack', 'solve_time']
    keys = summary_functions.keys()

    to_delete = []
    for k in keys:
        if k not in function_list:
            to_delete.append(k)
    for k in to_delete:
            del summary_functions[k]

def read_solution_data(case_folder, test_model, data_generator=tu.thermal_viol):
    parent, case = os.path.split(case_folder)
    ## assumed that detailed data is only needed for the nominal demand case
    filename = case + "_" + test_model + "_1000.json"

    try:
        md_dict = json.load(open(os.path.join(case_folder, filename)))
    except:
        return pd.DataFrame(data=None, index=[test_model])

    md = ModelData(md_dict)
    data = data_generator(md)
    cols = list(data.keys())
    new_cols = [int(c) for c in cols]
    df_data = pd.DataFrame(data, index=[test_model])
    df_data.columns = new_cols

    return df_data

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
        row = row[~np.isnan(row)]
        n = len(row)
        if n>0:
            sum = 0
            for i in range(n):
                sum += math.log(row[i])
            sum = sum / n
            gm = math.exp(sum)
            geomean.append(gm)
        else:
            geomean.append(np.nan)

    return geomean


def generate_summary_data(test_case, test_model_list, shorten=False):

    if shorten:
        short_summary()

    case_location = tau.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    ## include acopf results
    if 'acopf' not in test_model_list:
        test_model_list.append('acopf')

    df_data = pd.DataFrame(data=None, index=test_model_list)

    for func_name, func_dict in summary_functions.items():
        func = func_dict['function']
        summarizers = func_dict['summarizers']
        ## put data into blank DataFrame
        df_func = pd.DataFrame(data=None)

        for test_model in test_model_list:
            # read data and place in df_func
            df_raw = read_sensitivity_data(case_location, test_model, data_generator=func)
            df_func = pd.concat([df_func , df_raw], sort=True)

        ## also calculate geomean and maximum if function is solve_time()
        if 'geomean' in summarizers:
            gm = geometricMean(df_func.to_numpy())
            gm_name = func_name + '_geomean'
            df_gm = pd.DataFrame(data=gm, index=df_func.index, columns=[gm_name])
            df_data[gm_name] = df_gm

        if 'max' in summarizers:
            max = df_func.max(axis=1)
            max_name = func_name + '_max'
            df_max = pd.DataFrame(data=max.values, index=df_func.index, columns=['max_' + func_name])
            df_data[max_name] = df_max

        if 'avg' in summarizers:
            avg = df_func.mean(axis=1)
            if 'num' not in func_name:
                avg_name = func_name + '_avg'
            else:
                avg_name = func_name
            df_avg = pd.DataFrame(data=avg.values, index=df_func.index, columns=[func_name])
            df_data[avg_name] = df_avg

        if 'sum' in summarizers:
            sum = df_func.sum(axis=1)
            sum_name = func_name + '_sum'
            df_sum = pd.DataFrame(data=sum.values, index=df_func.index, columns=[func_name])
            df_data[sum_name] = df_sum

    ## save DATA to csv
    destination = tau.get_summary_file_location('data')
    filename = "summary_data_" + case_name + ".csv"
    df_data.to_csv(os.path.join(destination, filename))

def generate_violation_data(test_case, test_model_list, data_generator=None):

    case_location = tau.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    if data_generator is None:
        data_generator = tu.thermal_viol
    func_name = data_generator.__name__

    df_data = pd.DataFrame(data=None)

    for test_model in test_model_list:
        # read data and place in df_func
        try:
            df_raw = read_solution_data(case_location, test_model, data_generator=data_generator)
        except:
            df_raw = pd.DataFrame(data=None, index=[test_model])

        df_data = pd.concat([df_data, df_raw], sort=True)

    df_data = df_data.transpose()
    #df_data = df_data.sort_index(axis='columns')

    ## save DATA to csv
    destination = tau.get_summary_file_location('data')
    filename = func_name + '_' + case_name + ".csv"
    print('...out: {}'.format(filename))
    df_data.to_csv(os.path.join(destination, filename))

def generate_violation_heatmap(test_case, test_model_dict=None, viol_name=None, index_name=None, units=None,
                               colormap=None, show_plot=False):

    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    if viol_name is None or viol_name is 'thermal_viol':
        viol_name = 'thermal_viol'
        colormap = ListedColormap(colormap(np.linspace(0.5, 1, 256)))
        colormap.set_bad('grey')
    if index_name is None:
        index_name = 'Branch'
    if units is None:
        units = 'MW'

    filename = viol_name + "_" + case_name + ".csv"
    df_data = get_data(filename, test_model_dict=test_model_dict)

    data = df_data.values
    vmin = data.min()
    vmax = data.max()

    kwargs={}
    cbar_dict = {}
    cbar_dict['label'] = viol_name + ' (' + units + ')'
    if viol_name is 'vm_viol':
        kwargs['vmin'] = min(vmin,-vmax)
        kwargs['vmax'] = max(vmax,-vmin)
    kwargs['linewidth'] = 0
    kwargs['cmap'] = colormap
    kwargs['cbar_kws'] = cbar_dict

    # Create heatmap in Seaborn
    ## Create plot
    plt.figure(figsize=(4, 8.5))
    ax = sns.heatmap(df_data, **kwargs)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    ax.set_title(case_name + " " + viol_name)
    ax.set_xlabel("Model")
    ax.set_ylabel(index_name)

    plt.tight_layout()

    ## save FIGURE as png
    filename = case_name + "_" + viol_name + ".png"
    destination = tau.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()
        plt.clf()

def generate_sum_data(test_case, test_model_list, function_list=sum_functions):

    case_location = tau.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    ## include acopf results
    if 'acopf' not in test_model_list:
        test_model_list.append('acopf')

    df_data = pd.DataFrame(data=None, index=test_model_list)

    for func in function_list:
        ## put data into blank DataFrame
        df_func = pd.DataFrame(data=None)

        for test_model in test_model_list:
            # read data and place in df_func
            df_raw = read_sensitivity_data(case_location, test_model, data_generator=func)
            df_func = pd.concat([df_func , df_raw], sort=True)

        func_name = func.__name__

        df_func = df_func.sum(axis=1)
        df_func = pd.DataFrame(data=df_func.values, index=df_func.index, columns=[func_name])

        #df_data = pd.concat([df_data, df_func])
        df_data[func_name] = df_func

    ## save DATA to csv
    destination = tau.get_summary_file_location('data')
    filename = "sum_data_" + case_name + ".csv"
    df_data.to_csv(os.path.join(destination, filename))


def generate_speedup_data(case_list=None, mean_data='solve_time_geomean', benchmark='dlopf_lazy'):

    ## get data
    if case_list is None:
        case_list = tau.case_names[:]

    data_dict = {}
    cases = []
    for case in case_list:
        try:
            input = "summary_data_" + case + ".csv"
            df_data = get_data(input)
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
    destination = tau.get_summary_file_location('data')
    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data.to_csv(os.path.join(destination, filename))


def generate_speedup_heatmap(test_model_dict=None, mean_data='solve_time_geomean', benchmark='dlopf_lazy',colormap=None,
                             cscale='linear', include_benchmark=False, show_plot=False):

    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data = get_data(filename, test_model_dict=test_model_dict)
    if not include_benchmark and benchmark in df_data.columns.to_list():
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
    destination = tau.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))

    if show_plot:
        plt.show()
    else:
        plt.cla()
        plt.clf()


def generate_sensitivity_data(test_case, test_model_list, data_generator=tu.acpf_slack,
                              data_is_pct=False, data_is_vector=False, vector_norm=2):

    case_location = tau.get_solution_file_location(test_case)
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

    # iterate over test_models
    if 'acopf'  not in test_model_list:
        test_model_list.append('acopf')

    for test_model in test_model_list:
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
    destination = tau.get_summary_file_location('data')
    filename = "sensitivity_data_" + case_name + "_" + y_axis_data + ".csv"
    df_data.to_csv(os.path.join(destination, filename))



def get_data(filename, test_model_dict=None):
    print(filename)

    ## get data from CSV
    source = tau.get_summary_file_location('data')
    df_data = pd.read_csv(os.path.join(source,filename), index_col=0)

    if test_model_dict is not None:
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
    input = "summary_data_" + case_name + ".csv"
    df_data = get_data(input, test_model_dict=test_model_dict)

    models = list(df_data.index.values)
    df_y_data = df_data[y_data]
    df_x_data = df_data[x_data]

    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    markers = ['+','o','x','s']
    n = len(df_data)
    m = len(markers)
    new_marks = [markers[i%m] for i in range(n)]
    o_marks = [i for i in range(0,len(new_marks)) if new_marks[i] is 'o']
    edgecolor = [colors(i) for i in np.linspace(0, 1, n)]
    facecolor = edgecolor.copy()
    for idx in o_marks:
        facecolor[idx] = 'none'
    #if colors is not None:
    #    ax.set_prop_cycle(color=facecolor,
    #                      edgecolor=edgecolor,
    #                      marker=new_marks)

    for m in models:
        idx = models.index(m)
        edge = edgecolor[idx]
        face = facecolor[idx]
        marker_style = dict(linestyle='', markersize=8, color=edge)
        x = df_x_data[m]
        y = df_y_data[m]
        if m == 'acopf':
            #ax.scatter(x, y, s=mark_size, label=m, marker=mark_acopf)
            ax.plot(x,y, label=m, marker=mark_acopf, **marker_style)
        else:
            mark = new_marks[idx]
            #ax.scatter(x, y, s=mark_size, label=m, marker=mark, edgecolors=edge, facecolors=face)
            ax.plot(x,y, label=m, marker=mark, markeredgewidth=2, fillstyle='none', **marker_style)

        if annotate_plot:
            ax.annotate(m, (x,y))

    ax.set_title(y_data + " vs. " + x_data + "\n(" + case_name + ")")
    ax.set_ylabel(y_data + " (" + y_units + ")")
    ax.set_xlabel(x_data + " (" + x_units + ")")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ## save FIGURE to png
    figure_dest = tau.get_summary_file_location('figures')
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
    df_data = get_data(input, test_model_dict=test_model_dict)

    models = list(df_data.columns.values)


    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    markers = ['+','o','x','s']
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
            ax.plot(y, label=m, fillstyle='none')


    ax.set_title(plot_data + " (" + case_name + ")")
    # output.set_ylim(top=0)
    ax.set_xlabel("Demand Multiplier")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel(plot_data + " (" +  units + ")")

    ## save FIGURE as png
    destination = tau.get_summary_file_location('figures')
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
        case_list = tau.case_names[:]
    if s_data is None:
        var_names = ['model',x_data,y_data]
    else:
        var_names = ['model',x_data,y_data,s_data]
    sns_data = pd.DataFrame(data=None,columns=var_names)
    cases = []
    for case in case_list:
        try:
            input = "summary_data_" + case + ".csv"
            df_data = get_data(input, test_model_dict=test_model_dict)
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
    figure_dest = tau.get_summary_file_location('figures')
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
        case_list = tau.case_names[:]
    y_dict = {}
    x_dict = {}
    s_dict = {}
    if s_data is None:
        var_names = ['model',x_data,y_data]
    else:
        var_names = ['model',x_data,y_data,s_data]
    for case in case_list:
        try:
            input = "summary_data_" + case + ".csv"
            df_data = get_data(input, test_model_dict=test_model_dict)
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
    figure_dest = tau.get_summary_file_location('figures')
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

def sensitivity_plot(test_case, test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('cubehelix')

    sensitivity_dict = tau.get_sensitivity_dict(test_model_list)
    generate_sensitivity_data(test_case, test_model_list, data_generator=tu.acpf_slack)
    generate_sensitivity_plot(test_case, sensitivity_dict, plot_data='acpf_slack', units='MW', colors=colors, show_plot=show_plot)

def pareto_plot(test_case, test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('cubehelix')

    pareto_dict = tau.get_pareto_dict(test_model_list)
    generate_pareto_plot(test_case, pareto_dict, y_data='acpf_slack_avg', x_data='solve_time_geomean', y_units='MW', x_units='s',
                         mark_default='o', mark_lazy='+', mark_acopf='*', mark_size=100, colors=colors,
                         annotate_plot=False, show_plot=show_plot)

def solution_time_plot(test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('cubehelix')

    case_size_dict = tau.get_case_size_dict(test_model_list)
    generate_case_size_plot(case_size_dict, y_data='solve_time_geomean', y_units='s',
                            x_data='num_buses', x_units=None, s_data='con_per_bus',colors=colors,
                            xscale='log', yscale='linear',show_plot=show_plot)

def lazy_speedup_plot(test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('inferno')

    lazy_speedup_dict = tau.get_lazy_speedup_dict(test_model_list)
    generate_speedup_data(mean_data='solve_time_geomean', benchmark='acopf')
    generate_speedup_heatmap(test_model_dict=lazy_speedup_dict, mean_data='solve_time_geomean', benchmark='acopf',
                             include_benchmark=False, colormap=colors, cscale='log', show_plot=show_plot)

def trunc_speedup_plot(test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('inferno')

    trunc_speedup_dict = tau.get_trunc_speedup_dict(test_model_list)
    generate_speedup_data(mean_data='solve_time_geomean', benchmark='dlopf_default')
    generate_speedup_heatmap(test_model_dict=trunc_speedup_dict, mean_data='solve_time_geomean', benchmark='dlopf_default',
                             cscale='linear', include_benchmark=True, colormap=colors, show_plot=show_plot)

def acpf_violations_plot(test_case, test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('coolwarm')

    violation_dict = tau.get_violation_dict(test_model_list)
    generate_violation_data(test_case, test_model_list,data_generator=tu.thermal_viol)
    generate_violation_data(test_case, test_model_list,data_generator=tu.vm_viol)
    generate_violation_heatmap(test_case, test_model_dict=violation_dict,viol_name='thermal_viol',
                               index_name='Branch',colormap=colors,show_plot=show_plot)
    generate_violation_heatmap(test_case, test_model_dict=violation_dict,viol_name='vm_viol',
                               index_name='Bus',colormap=colors,show_plot=show_plot)

def create_full_summary(test_case, test_model_list, show_plot=True):
    """
    solves generates plots for test_case
    """

    colors = get_colors(map_name='cubehelix', trim=0.8)
    speed_colors = get_colors(map_name='inferno')
    viol_colors = get_colors(map_name='coolwarm')

    ## Generate data files
    generate_summary_data(test_case,test_model_list, shorten=False)

    ## Generate plots

    acpf_violations_plot(test_case, test_model_list, colors=viol_colors, show_plot=show_plot)

    sensitivity_plot(test_case, test_model_list, colors=colors, show_plot=show_plot)

    pareto_plot(test_case, test_model_list, colors=colors, show_plot=show_plot)

    solution_time_plot(test_model_list, colors=colors, show_plot=show_plot)

    lazy_speedup_plot(test_model_list, colors=speed_colors, show_plot=show_plot)

    trunc_speedup_plot(test_model_list, colors=speed_colors, show_plot=show_plot)


if __name__ == '__main__':
    import sys
    try:
        test_case = tau.idx_to_test_case(sys.argv[1])
    except:
        test_case = tau.idx_to_test_case(0)

    test_model_list = [
         'acopf',
         'slopf',
         'dlopf_default',
         'dlopf_lazy',
         'dlopf_e5',
         'dlopf_e4',
         'dlopf_e3',
         #'dlopf_e2',
         'clopf_default',
         'clopf_lazy',
         'clopf_e5',
         'clopf_e4',
         'clopf_e3',
         #'clopf_e2',
         'clopf_p_default',
         'clopf_p_lazy',
         'clopf_p_e5',
         'clopf_p_e4',
         'clopf_p_e3',
         #'clopf_p_e2',
         'qcopf_btheta',
         'dcopf_ptdf_default',
         'dcopf_ptdf_lazy',
         'dcopf_ptdf_e5',
         'dcopf_ptdf_e4',
         'dcopf_ptdf_e3',
         #'dcopf_ptdf_e2',
         'dcopf_btheta'
         ]

    #acpf_violations_plot(test_case,test_model_list)
    create_full_summary(test_case,test_model_list)
