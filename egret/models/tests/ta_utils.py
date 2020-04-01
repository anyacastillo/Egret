#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains some helpers and data for test_approximations
"""
import os

case_names = ['pglib_opf_case3_lmbd',
              'pglib_opf_case5_pjm',
              'pglib_opf_case14_ieee',
              'pglib_opf_case24_ieee_rts',
              'pglib_opf_case30_as',
              'pglib_opf_case30_fsr',
              'pglib_opf_case30_ieee',
              'pglib_opf_case39_epri',
              'pglib_opf_case57_ieee',
              'pglib_opf_case73_ieee_rts',
              'pglib_opf_case89_pegase', ### not feasible at mult = 1.01 ###
              'pglib_opf_case118_ieee',
              'pglib_opf_case162_ieee_dtc',
              'pglib_opf_case179_goc',
              'pglib_opf_case200_tamu',
              'pglib_opf_case240_pserc',
              'pglib_opf_case300_ieee',
              'pglib_opf_case500_tamu',
              'pglib_opf_case588_sdet',
              'pglib_opf_case1354_pegase',
              'pglib_opf_case1888_rte',
              'pglib_opf_case1951_rte',
              'pglib_opf_case2000_tamu',
              'pglib_opf_case2316_sdet',
              'pglib_opf_case2383wp_k',
              'pglib_opf_case2736sp_k',
              'pglib_opf_case2737sop_k',
              'pglib_opf_case2746wop_k',
              'pglib_opf_case2746wp_k',
              'pglib_opf_case2848_rte',
              'pglib_opf_case2853_sdet',
              'pglib_opf_case2868_rte',
              'pglib_opf_case2869_pegase',
              'pglib_opf_case3012wp_k',
              'pglib_opf_case3120sp_k',
              'pglib_opf_case3375wp_k',
              'pglib_opf_case4661_sdet',
              'pglib_opf_case6468_rte',
              'pglib_opf_case6470_rte',
              'pglib_opf_case6495_rte',
              'pglib_opf_case6515_rte',
              'pglib_opf_case9241_pegase',
              'pglib_opf_case10000_tamu',
              'pglib_opf_case13659_pegase',
              ]

test_cases = [os.path.join('../../../download/pglib-opf-master/', f + '.m') for f in case_names]

def idx_to_test_case(s):
    try:
        idx = int(s)
        tc = test_cases[idx]
        return tc
    except IndexError:
        raise SyntaxError("Index out of range of test_cases.")
    except ValueError:
        try:
            idx = case_names.index(s)
            tc = test_cases[idx]
            return tc
        except ValueError:
            raise SyntaxError(
                "Expecting argument of either A, B, C, D, E, or an index or case name from the test_cases list.")

def get_solution_file_location(test_case):
    _, case = os.path.split(test_case)
    case, _ = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    solution_location = os.path.join(current_dir, 'transmission_test_instances', 'approximation_solution_files', case)

    return solution_location

def get_summary_file_location(folder):
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    location = os.path.join(current_dir, 'transmission_test_instances','approximation_summary_files', folder)

    if not os.path.exists(location):
        os.makedirs(location)

    return location
