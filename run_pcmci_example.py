#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tigramite causal discovery for time series: Parallization script implementing 
the PCMCI method based on mpi4py. 

Parallelization is done across variables j for both the PC condition-selection
step and the MCI step.
"""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

# Angel Vázquez
# Period 1948-2011
# CMIP5 models
#

from mpi4py import MPI # ERROR https://stackoverflow.com/questions/36156822/error-when-starting-open-mpi-in-mpi-init-via-python
import numpy
import os, sys, cPickle, time
from datetime import datetime, date

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import pandas as pd

# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    #return [container[_i::count] for i in range(count)]
    return [container[i::count] for i in range(count)]


def run_pc_stable_parallel(j, dataframe, cond_ind_test, params):
    """Wrapper around PCMCI.run_pc_stable estimating the parents for a single 
    variable j.

    Parameters
    ----------
    j : int
        Variable index.

    Returns
    -------
    j, pcmci_of_j, parents_of_j : tuple
        Variable index, PCMCI object, and parents of j
    """

    N = dataframe.values.shape[1]

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        selected_variables=[j],
        # var_names=var_names,
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    if method_arg == 'pcmci':
        parents_of_j = pcmci_of_j.run_pc_stable(
                          selected_links=params['selected_links'],
                          tau_max=params['tau_max'],
                          pc_alpha=params['pc_alpha'],
                )
    elif method_arg == 'gc':
        parents_of_j = {}
        for i in range(N):
            if i == j:
                parents_of_j[i] = [(var, -lag)
                                 for var in range(N)
                                 for lag in range(params['tau_min'], params['tau_max'] + 1)
                                 ]
            else:
                parents_of_j[i] = []
    elif method_arg == 'corr':
        parents_of_j = {}
        for i in range(N):
            parents_of_j[i] = []

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


def run_mci_parallel(j, pcmci_of_j, all_parents, params):
    """Wrapper around PCMCI.run_mci step.

    Parameters
    ----------
    j : int
        Variable index.

    pcmci_of_j : object
        PCMCI object for variable j. This may contain pre-computed results 
        (such as residuals or null distributions).

    all_parents : dict
        Dictionary of parents for all variables. Needed for MCI independence
        tests.

    Returns
    -------
    j, results_in_j : tuple
        Variable index and results dictionary containing val_matrix, p_matrix,
        and optionally conf_matrix with non-zero entries only for
        matrix[:,j,:].
    """

    if method_arg == 'pcmci':
        results_in_j = pcmci_of_j.run_mci(
                selected_links=params['selected_links'],
                tau_min=params['tau_min'],
                tau_max=params['tau_max'],
                parents=all_parents,
                max_conds_px=params['max_conds_px'],
            )
    elif method_arg == 'gc':
        results_in_j = pcmci_of_j.run_mci(
                selected_links=params['selected_links'],
                tau_min=params['tau_min'],
                tau_max=params['tau_max'],
                parents=all_parents,
                max_conds_px=0,
            )
    elif method_arg == 'corr':
        results_in_j = pcmci_of_j.run_mci(
                selected_links=params['selected_links'],
                tau_min=params['tau_min'],
                tau_max=params['tau_max'],
                parents=all_parents,
                max_conds_py=0,
                max_conds_px=0,
            )

    return j, results_in_j


# period_length = int(sys.argv[1])           # period_length = 30, 60, 90, 120, 150, 180, 210
# n_comps = int(sys.argv[2])          # n_comps = 20, 60, 100

# JAKOB: Based on the full model time available, we chunk up the time axis into
# as many periods of length "length" (in years) we can fit into the full model time

period_length = 70
n_comps = 100

verbosity = 0

time_bin_length = 3
months = [12, 1, 2]                 # 



# calendar = "standard" # "365_DAY" "standard" "proleptic_gregorian"
# if calendar == "365_DAY":
#     model_name = ['inmcm4','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR']
# else:
# model_name = ['CNRM-CM5','MIROC-ESM-CHEM','MIROC-ESM','MRI-CGCM3', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM-P']
# model_name = ['xizka_selected_daily_150yrs.nc']
model_name = [
            # 'MIROC5_historical_r1i1p1_19480101-20180302.nc',
            # 'HadGEM2-ES_historical_r1i1p1_18591201-20180302.nc',
            # 'MIROC-ESM_historical_r1i1p1_19480101-20180302.nc',
            # 'MIROC-ESM-CHEM_historical_r1i1p1_19480101-20180302.nc',
            # 'MRI-CGCM3_historical_r1i1p1_19480101-20180302.nc',
            # 'MPI-ESM-MR_historical_r1i1p1_19480101-20180302.nc',
            # 'MPI-ESM-LR_historical_r1i1p1_19480101-20180302.nc',
            # 'GFDL-ESM2M_historical_r1i1p1_19480101-20180305.nc',
            # 'GFDL-ESM2G_historical_r1i1p1_18610101-20180302.nc',
            # 'psl_day_MPI-ESM-LR_historical_r1i1p1_19480101-20180302_three_members_detrend.nc',
            # 'SLP_ncar.nc',
            # 'CanESM2_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'CCSM4_historical_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'CSIRO-Mk3-6-0_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'EC-EARTH_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'HadGEM2-ES_rcp85_r1i1p1_19480101-20171230_detrend_3members.nc',
            # 'IPSL-CM5A-LR_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'MIROC5_historical_r1i1p1_19480101-20171231_detrend_3members.nc',
            # 'MPI-ESM-LR_historical_r1i1p1_19480101-20180302_three_members_detrend.nc',
            # 'hadgem2es_daily_slp_piControl_230yrs.nc',            
    'psl_day_CNRM-CM5_r1i1p1_19360101-20051231_detrend_10members_cat_regridded.nc',
]
# model_name_dict = {
#             'SLP_ncar.nc':'ncar',
#             'CanESM2_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc':'CanESM2',
#             'CCSM4_historical_r1i1p1_19480101-20171231_detrend_3members.nc':'CCSM4',
#             'CSIRO-Mk3-6-0_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc':'CSIRO-Mk360',
#             'EC-EARTH_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc':'EC-EARTH',
#             'psl_day_HadGEM2-ES_rcp85_r1i1p1_19480101-20171230_detrend_4members.nc':'HadGEM2-ES',
#             'psl_day_HadGEM2-CC_historical_r1i1p1_19480101-20171230_detrend_3ensemble_members_cat.nc':'HadGEM2-CC',
#             'psl_day_ACCESS1-3_historical_r1i1p1_19480101-20171231_detrend_3ensemble_members_cat.nc':'ACCESS1-3',
#             'psl_day_bcc-csm1-1_historical_r1i1p1_19480101-20171231_detrend_3ensemble_members_cat.nc':'BCC-CSM1-1',
#             'IPSL-CM5A-LR_rcp85_r1i1p1_19480101-20171231_detrend_3members.nc':'IPSL-CM5A-LR',
#             'MIROC5_historical_r1i1p1_19480101-20171231_detrend_3members.nc':'MIROC5',
#             'MPI-ESM-LR_historical_r1i1p1_19480101-20180302_three_members_detrend.nc':'MPI-ESM-LR',
#             'hadgem2es_daily_slp_piControl_230yrs.nc':'MOHC-pi',            
#             }

selected_components=[]
for i in range(1,51):
    selected_components.append('c'+str(i))
print selected_components

if os.path.expanduser('~') == '/home/rung_ja':
    comps_order_file=pd.DataFrame.from_csv(
                    '/home/rung_ja/Zwischenergebnisse/causal_robustness_cmip/selected_components_10_of_100.csv')
elif os.path.expanduser('~') == '/home/peer':
    comps_order_file=pd.DataFrame.from_csv(
                            '/home/peer/Documents/analysis_many_members/selected_components_10_of_100_manymembershist.csv')
elif os.path.expanduser('~') == '/home/pjn':
    comps_order_file=pd.DataFrame.from_csv(
                            '/home/pjn/Documents/Imperial/new_coll_Jakob/analysis_many_members/selected_comps_NCEP.csv')
else:
    comps_order_file=pd.DataFrame.from_csv('/rds/general/user/pnowack/home/cmip5_jakob/pcmci_3dm_varimax/50_comps_NCEP_varimax/selected_comps_NCEP2.csv')


print("Selected models: " + str(model_name))
for model in model_name:
    for method_arg in ['pcmci']:

        print("Setup %s %s" %(model, method_arg))
        #datadict = cPickle.load(open( os.path.expanduser('~/work/projects/Autocorrelations/data/pressure_components.pkl'), 'rb'))
        # datadict = cPickle.load(open('/home/angelv/Dropbox/PhD/runge/WorkSpace/varimax_components/'+model+'_slp_detrended_cosweights_daily_aggregated.bin', 'rb'))
        if os.path.expanduser('~') == '/home/rung_ja':
            # file_name = '/home/jakobrunge/Daten/varimax_components/peer/varimax_%s_comps-%d_months-%s.bin' % (model, n_comps, months)
            # file_name = '/media/peer/Firecuda/cmip5_jakob/varimax_pcmci_3dm_data_and_figures/'+str(model_name_dict[model])+'/varimax_%s_3dm_comps-%d_months-%s.bin' % (model, n_comps, months)
            file_name = '/home/rung_ja/Zwischenergebnisse/causal_robustness_cmip/varimax_%s_3dm_comps-%d_months-%s.bin' % (model, n_comps, months)
        elif os.path.expanduser('~') == '/home/peer':
            file_name = '/home/peer/Documents/analysis_many_members/varimax/varimax_data/varimax_%s_3dm_comps-%d_months-%s.bin' % (model, n_comps, months)
        elif os.path.expanduser('~') == '/home/pjn':
            file_name = '/home/pjn/Documents/Imperial/new_coll_Jakob/analysis_many_members/varimax/varimax_data/varimax_%s_3dm_comps-%d_FW_months-%s.bin' % (model, n_comps, months)
        else:
            # file_name = '/media/peer/Firecuda/cmip5_jakob/varimax_pcmci_3dm_data_and_figures/'+str(model_name_dict[model])+'/varimax_%s_3dm_comps-%d_months-%s.bin' % (model, n_comps, months)
            file_name = '/rds/general/project/nowack_graven/live/projects_jakob/model_eval/pre_sub/varimax_3dm/comps_like_NCEP'+'/varimax_%s_3dm_comps-%d_FW_months-%s.bin' % (model, n_comps, months)
            # file_name = '/home/jakrunge/Daten/varimax_components/peer/varimax_%s_comps-%d_months-%s.bin' % (model, n_comps, months)

        datadict = cPickle.load(open(file_name, 'rb'))

        # print datadict
        # continue

        d = datadict['results']
        time_mask = d['time_mask']
        dateseries = d['time'][:]

        # print d['start_end'][1]
        # new_end
        # print dateseries
        # sys.exit(0)

        # Compute a time_axis from the start and end dates, 
        # converted to years with decimals
        # n_days=360.
        # correct_start = float(d['start_end'][0].split(' ')[0].split('-')[0]) \
        #                 + (float(d['start_end'][0].split(' ')[0].split('-')[1]) - 1.)/12. \
        #                 + (float(d['start_end'][0].split(' ')[0].split('-')[2]) - 1.)/n_days
        # correct_end = float(d['start_end'][1].split(' ')[0].split('-')[0]) \
        #                 + (float(d['start_end'][1].split(' ')[0].split('-')[1]) - 1.)/12. \
        #                 + (float(d['start_end'][1].split(' ')[0].split('-')[2]) - 1.)/n_days

        # print correct_start, correct_end
        # print 'model sample length: ',len(d['time'])
        # n_ensembles=4
        # new_end = correct_start + (correct_end - correct_start)*n_ensembles
        # # time_axis = numpy.linspace(correct_start, new_end, len(d['time']))

        # # else:
        # #     time_axis = numpy.linspace(correct_start, correct_end, len(d['time']))


        # # time_axis = numpy.linspace(new_start, 
        # #                         new_end, 
        # #                         len(d['time']))
        # # # time_axis = numpy.linspace(int(d['start_end'][0].split('-')[0]), 
        # #                         int(d['start_end'][1].split('-')[0])+1, 
        # # #                         len(d['time']))
        # # print d['start_end']
        # # print time_axis,', time axis of length ', len(time_axis)

        # # JAKOB: Based on the full model time available, we chunk up the time axis into
        # # as many periods of length "length" (in years) we can fit into the full model time
        # # model_sample_length = len(d['time'])
        # # period_sample_length = model_sample_length - numpy.argmax(time_axis > (time_axis[-1] - period_length))
        period_indices=[0,25567,51135,76703,102271,127839,153407,178975,204543,230111,255679]
        # period_indices = range(model_sample_length-1, 0, -period_sample_length)[::-1]
        print "period_indices ", period_indices
        # print 
        for ip, period_start_index in enumerate(period_indices[:-1]):
            period_end_index = period_indices[ip+1]
            print period_start_index, period_end_index, period_end_index-period_start_index #, time_axis[period_start_index], time_axis[period_end_index]

            # continue


            fulldata = d['ts_unmasked']
            N = fulldata.shape[1]
            fulldata_mask = numpy.repeat(time_mask.reshape(len(d['time']), 1), N,  axis=1)


            fulldata = fulldata[period_start_index:period_end_index, :]
            fulldata_mask = fulldata_mask[period_start_index:period_end_index, :]

            print fulldata.shape



            print("Fulldata shape = %s" % str(fulldata.shape))
            print("Fulldata masked shape = %s" % str(fulldata_mask.shape))
            print("Unmasked samples %d" % (fulldata_mask[:,0]==False).sum())

            print("Aggregating data to time_bin_length=%s" %time_bin_length)

            ## Time bin data
            fulldata = pp.time_bin_with_mask(fulldata, time_bin_length=time_bin_length)[0]
            fulldata_mask = pp.time_bin_with_mask(fulldata_mask, time_bin_length=time_bin_length)[0] > 0.
            print("Fulldata after binning shape = %s" % str(fulldata.shape))
            print("Fulldata after binning masked shape = %s" % str(fulldata_mask.shape))
            
            # Only use selected indices
            selected_comps_indices=[]
            for i in selected_components:
                selected_comps_indices.append(int(comps_order_file['comps'][i]))
            
            fulldata = fulldata[:, selected_comps_indices]
            fulldata_mask = fulldata_mask[:, selected_comps_indices]



            dataframe = pp.DataFrame(fulldata, mask=fulldata_mask)

          
            print("Fulldata shape = %s" % str(dataframe.values.shape))
            print("Unmasked samples %d" % (dataframe.mask[:,0]==False).sum())  
            
            # sys.exit(0)
            T, N = dataframe.values.shape
            #print(N)

            resdict = {
                "CI_params":{
                    'significance':'analytic', 
                    'use_mask':True,
                    'mask_type':['y'],
                    'recycle_residuals':False,
                    },

                "PC_params":{                    
                    # Significance level in condition-selection step. If a list of levels is is
                    # provided or pc_alpha=None, the optimal pc_alpha is automatically chosen via
                    # model-selection.
                    'pc_alpha':None,
                    # Minimum time lag (must be >0)
                    'tau_min':1,
                    # Maximum time lag
                    'tau_max':10,
                    # Maximum cardinality of conditions in PC condition-selection step. The
                    # recommended default choice is None to leave it unrestricted.
                    'max_conds_dim':None,
                    # Selected links may be used to restricted estimation to given links.
                    'selected_links':None,
                    'selected_variables' : range(N), #selected_comps_indices,
                    # Optionalonally specify variable names
                    # 'var_names':range(N),
                    'var_names': selected_comps_indices,
                    },

                "MCI_params":{
                    # Minimum time lag (can also be 0)
                    'tau_min':0,
                    # Maximum time lag
                    'tau_max':10,
                    # Maximum number of parents of X to condition on in MCI step, leave this to None
                    # to condition on all estimated parents.
                    'max_conds_px':None,
                    # Selected links may be used to restricted estimation to given links.
                    'selected_links':None,
                    # Alpha level for MCI tests (just used for printing since all p-values are 
                    # stored anyway)
                    'alpha_level' : 0.05,
                    }
                }

            # Chosen conditional independence test
            cond_ind_test = ParCorr(verbosity=verbosity, **resdict['CI_params'] )
                # significance='analytic', 
                # use_mask=True,
                # mask_type=['y'],
                # recycle_residuals=True,
                # verbosity=verbosity)
            
            # Store results in file
            if os.path.expanduser('~') == '/home/rung_ja':
                file_name = '/home/rung_ja/Zwischenergebnisse/causal_robustness_cmip/results_%s_comps-%d_months-%s_%s_%s_%s.bin' % (model, n_comps, months, method_arg, period_length, ip)
            elif os.path.expanduser('~') == '/home/peer':
                file_name = '/home/peer/Documents/analysis_many_members/pcmci/results/results_%s_comps-%d_months-%s_%s_%s_%s.bin' % (model, n_comps, months, method_arg, period_length, ip)
            elif os.path.expanduser('~') == '/home/pjn':
                file_name = '/home/pjn/Documents/Imperial/new_coll_Jakob/analysis_many_members/pcmci/results/results_%s_comps-%d_months-%s_%s_%s_%s.bin' % (model, n_comps, months, method_arg, period_length, ip)
            else:
                # file_name = '/p/projects/synchronet/Users/jakrunge/results/autocorrelations/cmip5/varimax_results_%s_comps-%d_months-%s_%s_%s_%s.bin' % (model, n_comps, months, method_arg, length, data_split)
                file_name = '/rds/general/project/nowack_graven/live/projects_jakob/model_eval/pre_sub/varimax_3dm/comps_like_NCEP/pcmci_results'+'/pcmci_results_%s_3dm_comps-%d_months-%s_%s_%s_%s.bin' % (model, n_comps, months, method_arg, period_length, ip)

            print(file_name)
            
            #
            #  Start of the script
            #
            if COMM.rank == 0:
            # if 3 == 0:

                # Only the master node (rank=0) runs this
                if verbosity > -1:
                    print("\n##\n## Running Parallelized Tigramite PC algorithm\n##"
                          "\n\nParameters:")
                    print("\nindependence test = %s" % cond_ind_test.measure
                          + "\ntau_min = %d" % resdict['PC_params']['tau_min']
                          + "\ntau_max = %d" % resdict['PC_params']['tau_max']
                          + "\npc_alpha = %s" % resdict['PC_params']['pc_alpha']
                          + "\nmax_conds_dim = %s" % resdict['PC_params']['max_conds_dim'])
                    print("\n")
            
                # Split selected_variables into however many cores are available.
                splitted_jobs = split(resdict['PC_params']['selected_variables'], COMM.size)
                if verbosity > -1:
                    print("Splitted selected_variables = "), splitted_jobs
            else:
                splitted_jobs = None
            
            
            ##
            ##  PC algo condition-selection step
            ##
            # Scatter jobs across cores.
            scattered_jobs = COMM.scatter(splitted_jobs, root=0)
            
            print("\nCPU %d estimates parents of %s" % (COMM.rank, scattered_jobs))
            
            # Now each rank just does its jobs and collects everything in a results list.
            results = []
            time_start = time.time()
            for j_index, j in enumerate(scattered_jobs):
                # Estimate conditions
                (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j, dataframe, cond_ind_test, params=resdict['PC_params'])
            
                results.append((j, pcmci_of_j, parents_of_j))
            
                num_here = len(scattered_jobs)
                current_runtime = (time.time() - time_start)/3600.
                current_runtime_hr = int(current_runtime)
                current_runtime_min = 60.*(current_runtime % 1.)
                estimated_runtime = current_runtime * num_here / (j_index+1.)
                estimated_runtime_hr = int(estimated_runtime)
                estimated_runtime_min = 60.*(estimated_runtime % 1.)
                # print ("\t# CPU %s task %d/%d: %dh %.1fmin / %dh %.1fmin: Variable %s" % (COMM.rank, j_index+1, num_here, 
                #                         current_runtime_hr, current_runtime_min, 
                #                         estimated_runtime_hr, estimated_runtime_min,  resdict['PC_params']['var_names'][j]))
            
            
            
            # Gather results on rank 0.
            results = MPI.COMM_WORLD.gather(results, root=0)
            
            
            if COMM.rank == 0:
                # Collect all results in dictionaries and send results to workers
                all_parents = {}
                pcmci_objects = {}
                for res in results:
                    for (j, pcmci_of_j, parents_of_j) in res:
                        all_parents[j] = parents_of_j[j]
                        pcmci_objects[j] = pcmci_of_j
                print(pcmci_objects[0].__dict__.keys())
                #if verbosity > -1:
                #    print("\n\n## Resulting condition sets:")
                #    for j in [var for var in all_parents.keys()]:
                #       pcmci_objects[j]._print_parents_single(j, all_parents[j],
                #           pcmci_objects[j].p_max[j],
                #           pcmci_objects[j].p_max[j]) ERROR IN GETTING p_max[] attribute
                        #pcmci_objects[j]._print_parents_single(j, all_parents[j],
                        #                        pcmci_objects[j].test_statistic_values[j], # ERROR
                        #                        pcmci_objects[j].p_max[j])
            
                if verbosity > -1:
                    print("\n##\n## Running Parallelized Tigramite MCI algorithm\n##"
                          "\n\nParameters:")
            
                    print("\nindependence test = %s" % cond_ind_test.measure
                            + "\ntau_min = %d" % resdict['MCI_params']['tau_min']
                            + "\ntau_max = %d" % resdict['MCI_params']['tau_max']
                          + "\nmax_conds_px = %s" % resdict['MCI_params']['max_conds_px'])
                    
                    print("Master node: Sending all_parents and pcmci_objects to workers.")
                
                for i in range(1, COMM.size):
                    COMM.send((all_parents, pcmci_objects), dest=i)
            
            else:
                if verbosity > -1:
                    print("Slave node %d: Receiving all_parents and pcmci_objects..."
                          "" % COMM.rank)
                (all_parents, pcmci_objects) = COMM.recv(source=0)
            
            
            ##
            ##   MCI step
            ##
            # Scatter jobs again across cores.
            scattered_jobs = COMM.scatter(splitted_jobs, root=0)
            
            # Now each rank just does its jobs and collects everything in a results list.
            results = []
            for j_index, j in enumerate(scattered_jobs):
                # print("\n\t# Variable %s (%d/%d)" % (var_names[j], j_index+1, len(scattered_jobs)))
                
                (j, results_in_j) = run_mci_parallel(j, pcmci_objects[j], all_parents, params=resdict['MCI_params'])
                results.append((j, results_in_j))
            
                num_here = len(scattered_jobs)
                current_runtime = (time.time() - time_start)/3600.
                current_runtime_hr = int(current_runtime)
                current_runtime_min = 60.*(current_runtime % 1.)
                estimated_runtime = current_runtime * num_here / (j_index+1.)
                estimated_runtime_hr = int(estimated_runtime)
                estimated_runtime_min = 60.*(estimated_runtime % 1.)
                # print ("\t# CPU %s task %d/%d: %dh %.1fmin / %dh %.1fmin: Variable %s" % (COMM.rank, j_index+1, num_here, 
                #                         current_runtime_hr, current_runtime_min, 
                #                         estimated_runtime_hr, estimated_runtime_min,  resdict['PC_params']['var_names'][j]))
            
            
            
            # Gather results on rank 0.
            results = MPI.COMM_WORLD.gather(results, root=0)
            
            
            if COMM.rank == 0:
                # Collect all results in dictionaries
                # 
                if verbosity > -1:
                    print("\nCollecting results...")
                all_results = {}
                for res in results:
                    for (j, results_in_j) in res:
                        for key in results_in_j.keys():
                            if results_in_j[key] is None:  
                                all_results[key] = None
                            else:
                                if key not in all_results.keys():
                                    if key == 'p_matrix':
                                        all_results[key] = numpy.ones(results_in_j[key].shape)
                                    else:
                                        all_results[key] = numpy.zeros(results_in_j[key].shape)
                                    all_results[key][:,j,:] =  results_in_j[key][:,j,:]
                                else:
                                    all_results[key][:,j,:] =  results_in_j[key][:,j,:]
            
            
                p_matrix=all_results['p_matrix']
                val_matrix=all_results['val_matrix']
                conf_matrix=all_results['conf_matrix']
            
                sig_links = (p_matrix <= resdict['MCI_params']['alpha_level'])
            
                if verbosity > -1:
                    print("\n## Significant links at alpha = %s:" % resdict['MCI_params']['alpha_level'])
                    for j in resdict['PC_params']['selected_variables']:
            
                        links = dict([((p[0], -p[1] ), numpy.abs(val_matrix[p[0], 
                                        j, abs(p[1])]))
                                      for p in zip(*numpy.where(sig_links[:, j, :]))])
            
                        # Sort by value
                        sorted_links = sorted(links, key=links.get, reverse=True)
            
                        n_links = len(links)
            
                        string = ""
                        string = ("\n    Variable %s has %d "
                                  "link(s):" % (resdict['PC_params']['var_names'][j], n_links))
                        for p in sorted_links:
                            string += ("\n        (%s %d): pval = %.5f" %
                                       (resdict['PC_params']['var_names'][p[0]], p[1], 
                                        p_matrix[p[0], j, abs(p[1])]))
            
                            string += " | val = %.3f" % (
                                val_matrix[p[0], j, abs(p[1])])
            
                            if conf_matrix is not None:
                                string += " | conf = (%.3f, %.3f)" % (
                                    conf_matrix[p[0], j, abs(p[1])][0], 
                                    conf_matrix[p[0], j, abs(p[1])][1])
            
                        print(string)
            
            
                if verbosity > -1:
                    print("Pickling to "), file_name

                resdict['results'] = all_results
                file = open(file_name, 'wb')
                cPickle.dump(resdict, file, protocol=-1)        
                file.close()
                # PCMCI._print_significant_links(
                #        p_matrix=all_results['p_matrix'],
                #        val_matrix=all_results['val_matrix'],
                #        alpha_level=alpha_level,
                #        conf_matrix=all_results['conf_matrix'])
    
