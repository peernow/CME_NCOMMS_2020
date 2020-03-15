"""Tigramite data processing functions."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


import numpy as np
import sys, warnings


from scipy.linalg import svdvals, svd
import scipy.stats as scist

import cPickle

from datetime import date, datetime

from geo_field_jakob import GeoField
import tigramite
from tigramite import data_processing as pp

# data_dir=sys.argv[1]
# filename=sys.argv[2]
# savefolder=sys.argv[3]
def load_data(load_filename, folder_name, varname, 
                from_date=None, to_date=None,
                anomalize=None, anomalize_variance=None,
                anomalize_base=None,
                slice_lat=None, slice_lon=None,
                # months=None, 
                level=None,
                use_cdftime=True,
                verbosity=0,):

    filename = folder_name + load_filename

    if verbosity > 0:
        print("Loading data:")


    geo_object = GeoField()
    geo_object.load(filename, varname, use_cdftime)
    if verbosity > 0:
        print("\tOriginal date range %s - %s " % (geo_object.start_date, geo_object.end_date))
        print("Original data shape %s" % (str(geo_object.data().shape)))

    if level is not None:
        if verbosity > 0:
            print("\tSlicing levels: %s" % level)
        geo_object.slice_level(level)

    if verbosity > 0:
        print("\tSlicing lon = %s, lat = %s" % (slice_lon, slice_lat))
    geo_object.slice_spatial(slice_lon, slice_lat)
    
    # print geo_object.data()

    if anomalize:
        if verbosity > 0:   
            print("\tanomalize %s with base period %s" % (anomalize, str(anomalize_base)))

        geo_object.transform_to_anomalies(anomalize_base,
                                          anomalize)
        # print geo_object.tm[0]
    # print geo_object.data()
    # if anomalize_variance:
    #     if verbosity > 0:
    #         print("\tanomalize variance with base period %s" % str(anomalize_base))
    #     geo_object.normalize_variance(anomalize_base)

    # print geo_object.d[::12].mean(axis=0)
    if from_date is not None and to_date is not None:
        if verbosity > 0:
            print("\tSlicing from = %s to %s" % (from_date, to_date))
        geo_object.slice_date_range(from_date, to_date)

    # if months is not None:
    #     if verbosity > 0:
    #         print("\tSlicing months %s" % months)
    #     geo_object.slice_months(months)
    
    # print geo_object.data[ :10, 0, 0]
    return geo_object

def preprocess_data(geo_object,
                    detrend=True, 
                    cos_reweighting=True,
                    verbosity=0):

    if verbosity > 0:
        print("Preprocessing:")

    if detrend:
        if verbosity > 0:
            print("\tdetrend")
        geo_object.detrend()

    if cos_reweighting:
        if verbosity > 0:
            print("\tcos_reweighting by sqrt(cos(lat))")
        geo_object.qea_latitude_weights()

    return geo_object


def svd_flip(u, v=None, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if v is None:
         # rows of v, columns of u
         max_abs_rows = np.argmax(np.abs(u), axis=0)
         signs = np.sign(u[max_abs_rows, xrange(u.shape[1])])
         u *= signs
         return u

    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]

    return u, v


def pca_svd(data,
    truncate_by='max_comps', 
    max_comps=60,
    fraction_explained_variance=0.9,
    verbosity=0,
    ):
    """Assumes data of shape (obs, vars).

    https://stats.stackexchange.com/questions/134282/relationship-between-svd-
    and-pca-how-to-use-svd-to-perform-pca

    SVD factorizes the matrix A into two unitary matrices U and Vh, and a 1-D
    array s of singular values (real, non-negative) such that A == U*S*Vh,
    where S is a suitably shaped matrix of zeros with main diagonal s.

    K = min (obs, vars)

    U are of shape (vars, K)
    Vh are loadings of shape (K, obs)

    """

    n_obs = data.shape[0]

    # Center data
    data -= data.mean(axis=0)

    # data_T = np.fastCopyAndTranspose(data)
    # print data.shape

    U, s, Vt = svd(data, 
                full_matrices=False)
                # False, True, True)

    # flip signs so that max(abs()) of each col is positive
    U, Vt = svd_flip(U, Vt, u_based_decision=False)

    V = Vt.T
    S = np.diag(s)   
    # eigenvalues of covariance matrix
    eig = (s ** 2) / (n_obs - 1.)

    # Sort
    idx = eig.argsort()[::-1]
    eig, U = eig[idx], U[:, idx]


    if truncate_by == 'max_comps':

        U = U[:, :max_comps]
        V = V[:, :max_comps]
        S = S[0:max_comps, 0:max_comps]
        explained = np.sum(eig[:max_comps]) / np.sum(eig)

    elif truncate_by == 'fraction_explained_variance':
        # print np.cumsum(s2)[:80] / np.sum(s2)
        max_comps = np.argmax(np.cumsum(eig) / np.sum(eig) > fraction_explained_variance) + 1
        explained = np.sum(eig[:max_comps]) / np.sum(eig)


        U = U[:, :max_comps]
        V = V[:, :max_comps]
        S = S[0:max_comps, 0:max_comps]

    else:
        max_comps = U.shape[1]
        explained = np.sum(eig[:max_comps]) / np.sum(eig)

    # Time series
    ts = U.dot(S)

    return V, U, S, ts, eig, explained, max_comps

def varimax(Phi, gamma = 1.0, q = 500, 
    rtol = np.finfo(np.float32).eps ** 0.5,
    verbosity=0):
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    # print Phi
    for i in range(q):
        if verbosity > 1:
            if i % 10 == 0.:
                print("\t\tVarimax iteration %d" % i)
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 
                   - (gamma/float(p)) * np.dot(Lambda, 
                    np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and abs(d - d_old) / d < rtol: break
    # print i
    return np.dot(Phi, R), R

def get_varimax_loadings(geo_object, month_mask=None,
                    truncate_by = 'max_comps', 
                    max_comps=60,
                    fraction_explained_variance=0.9,
                    verbosity=0,
                    ):

    
    if verbosity > 0:
        print("Get Varimax components")
        print("\tGet SVD")
    data = geo_object.data()
    print(data.shape,' data shape after loading')
    if month_mask is not None:
        if verbosity > 0:
            print("\tCompute covariance only from months %s" % month_mask +
                  "\n\t(NOTE: Time series will be all months and mask can be retrieved from dict)")

        masked_data, time_mask = geo_object.return_masked_months(month_mask)
    else:
        masked_data = data
        time_mask = np.zeros(data.shape[0], dtype='bool')
    print masked_data.shape,' masked data shape before reshape'
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:]))) # flattening field of daily data
    masked_data = np.reshape(masked_data, (masked_data.shape[0], np.prod(data.shape[1:]))) # flattening field of daily data
    print masked_data.shape,data.shape,'masked and data after reshape'
    masked_data, Tbin = pp.time_bin_with_mask(masked_data,time_bin_length=3)
    print masked_data.shape,' masked data shape after binning'
    # Get truncated SVD
    V, U, S, ts_svd, eig, explained, max_comps = pca_svd(masked_data, truncate_by=truncate_by, max_comps=max_comps,
                                   fraction_explained_variance=fraction_explained_variance,
                                    verbosity=verbosity)
        # if verbosity > 0:
        #     print("Explained variance at max_comps = %d: %.5f" % (max_comps, explained))

    if verbosity > 0:
        if truncate_by == 'max_comps':

            print("\tUser-selected number of components: %d\n"
                  "\tExplaining %.2f of variance" %(max_comps, explained))

        elif truncate_by == 'fraction_explained_variance':

            print("\tUser-selected explained variance: %.2f of total variance\n"
                  "\tResulting in %d components" %(explained, max_comps))


    if verbosity > 0:
        print("\tVarimax rotation")
    # Rotate
    Vr, Rot = varimax(V, verbosity=verbosity)
    # Vr = V
    # Rot = np.diag(np.ones(V.shape[1]))
    # print Vr.shape
    Vr = svd_flip(Vr)

    if verbosity > 0:
        print("\tFurther metrics")
    # Get explained variance of rotated components
    s2 = np.diag(S)**2 / (masked_data.shape[0] - 1.)

    # matrix with diagonal containing variances of rotated components
    S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    expvar = np.diag(S2r)

    sorted_expvar = np.sort(expvar)[::-1]
    # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # reorder all elements according to explained variance (descending)
    nord = np.argsort(expvar)[::-1]
    Vr = Vr[:, nord]

    # Get time series of UNMASKED data
    comps_ts = data.dot(Vr)

    comps_ts_masked = masked_data.dot(Vr)

    # Get location of absmax
    comp_loc = {'x':np.zeros(max_comps), 'y':np.zeros(max_comps)}
    for i in range(max_comps):
        coords = np.unravel_index(np.abs(Vr[:, i]).argmax(), 
            dims=(len(geo_object.lats), len(geo_object.lons)))
        comp_loc['x'][i] = geo_object.lons[coords[1]]
        comp_loc['y'][i] = geo_object.lats[coords[0]]

    total_var = np.sum(np.var(masked_data, axis = 0))

    # print time_mask
    # print expvar
    # start_end = (str(date.fromordinal(int(geo_object.tm[0]))),
    #                   str(date.fromordinal(int(geo_object.tm[-1]))))
    start_end = (str(geo_object.start_date), str(geo_object.end_date))

    # print start_end_year

    return {'weights' : np.copy(Vr), 
            'ts_unmasked':comps_ts,
            'ts_masked':comps_ts_masked,
            'explained_var':sorted_expvar,
            'unrotated_weights':V,
            'explained': explained,
            'pca_eigs':eig,

            'truncate_by' : truncate_by, 
            'max_comps':max_comps,
            'fraction_explained_variance':fraction_explained_variance,
            'total_var' : total_var,

            'month_mask':month_mask,
            'comps_max_loc': comp_loc,
            'time_mask' : time_mask,
            'start_end' : start_end,
            'time' : geo_object.tm,
            'lats' : geo_object.lats, 
            'lons' : geo_object.lons,
            }


def get_results_from_weights(geo_object, 
                        weights_filename,
                        verbosity=0):

    weights_dict_results = cPickle.load(open(weights_filename))['results']
    results = weights_dict_results.copy()
    weights = results['weights']
    month_mask = results['month_mask']
    max_comps = results['max_comps']
    truncate_by = results['truncate_by']
    fraction_explained_variance = results['fraction_explained_variance']

    if verbosity > 0:
        print("Get Varimax components from %s" % weights_filename)
    data = geo_object.data()
    print(data.shape,' data shape after loading')
    if month_mask is not None:
        if verbosity > 0:
            print("\tCompute covariance only from months %s" % month_mask +
                  "\n\t(NOTE: Time series will be all months and mask can be retrieved from dict)")

        masked_data, time_mask = geo_object.return_masked_months(month_mask)
    else:
        masked_data = data
        time_mask = np.zeros(data.shape[0], dtype='bool')
    print masked_data.shape,' masked data shape before reshape'
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:]))) # flattening field of daily data
    masked_data = np.reshape(masked_data, (masked_data.shape[0], np.prod(data.shape[1:]))) # flattening field of daily data
    print masked_data.shape,data.shape,'masked and data after reshape'
    masked_data, Tbin = pp.time_bin_with_mask(masked_data,time_bin_length=3)
    print masked_data.shape,' masked data shape after binning'
    # # Get truncated SVD
    # V, U, S, ts_svd, eig, explained, max_comps = pca_svd(masked_data, truncate_by=truncate_by, max_comps=max_comps,
    #                                fraction_explained_variance=fraction_explained_variance,
    #                                 verbosity=verbosity)
    #     # if verbosity > 0:
        #     print("Explained variance at max_comps = %d: %.5f" % (max_comps, explained))

    # if verbosity > 0:
    #     if truncate_by == 'max_comps':

    #         print("\tUser-selected number of components: %d\n"
    #               "\tExplaining %.2f of variance" %(max_comps, explained))

    #     elif truncate_by == 'fraction_explained_variance':

    #         print("\tUser-selected explained variance: %.2f of total variance\n"
    #               "\tResulting in %d components" %(explained, max_comps))


    # if verbosity > 0:
    #     print("\tVarimax rotation")
    # # Rotate
    # Vr, Rot = varimax(V, verbosity=verbosity)
    # # Vr = V
    # # Rot = np.diag(np.ones(V.shape[1]))
    # # print Vr.shape
    # Vr = svd_flip(Vr)

    # if verbosity > 0:
    #     print("\tFurther metrics")
    # # Get explained variance of rotated components
    # s2 = np.diag(S)**2 / (masked_data.shape[0] - 1.)

    # # matrix with diagonal containing variances of rotated components
    # S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
    # expvar = np.diag(S2r)

    # sorted_expvar = np.sort(expvar)[::-1]
    # # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

    # # reorder all elements according to explained variance (descending)
    # nord = np.argsort(expvar)[::-1]
    # Vr = Vr[:, nord]

    if verbosity > 0:
        print("\tCompute components using weights ")
    # Get time series of UNMASKED data
    comps_ts = data.dot(weights)

    # print comps_ts[:2]
    # print weights_dict_results['ts_unmasked'][:2]
    # assert np.allclose(comps_ts, weights_dict_results['ts_unmasked'])


    comps_ts_masked = masked_data.dot(weights)

    # Get location of absmax
    comp_loc = {'x':np.zeros(max_comps), 'y':np.zeros(max_comps)}
    for i in range(max_comps):
        coords = np.unravel_index(np.abs(weights[:, i]).argmax(), 
            dims=(len(geo_object.lats), len(geo_object.lons)))
        comp_loc['x'][i] = geo_object.lons[coords[1]]
        comp_loc['y'][i] = geo_object.lats[coords[0]]

    total_var = np.sum(np.var(masked_data, axis = 0))

    if verbosity > 0:
        print("\ttotal_var = ", total_var)
        print("\tSetting start_end from data_parameters")
    # print time_mask
    # print expvar
    # start_end = (str(date.fromordinal(int(geo_object.tm[0]))),
    #                   str(date.fromordinal(int(geo_object.tm[-1]))))
    start_end = (str(geo_object.start_date), str(geo_object.end_date))


    # Overwrite entries and delete those not needed anyway

    return {'weights' : weights, 
            'ts_unmasked':comps_ts,
            'ts_masked':comps_ts_masked,
            # 'explained_var':sorted_expvar,
            # 'unrotated_weights':V,
            # 'explained': explained,
            # 'pca_eigs':eig,

            'truncate_by' : truncate_by, 
            'max_comps':max_comps,
            'fraction_explained_variance':fraction_explained_variance,
            'total_var' : total_var,

            'month_mask':month_mask,
            'comps_max_loc': comp_loc,
            'time_mask' : time_mask,
            'start_end' : start_end,
            'time' : geo_object.tm,
            'lats' : geo_object.lats, 
            'lons' : geo_object.lons,
            }


def generate_pdf(dict, save_folder, save_name):

    import matplotlib
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    from scipy import signal
    import matplotlib.mlab as mlab
    from scipy import signal

    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams['xtick.labelsize'] = 7
    matplotlib.rcParams['ytick.labelsize'] = 7

    print("Plotting loadings and time series to %s.pdf" % (save_folder + save_name))

    d = dict['results']
    n_comps = d['max_comps']
    lons = d['lons']
    lats = d['lats']
    weights = d['weights'].reshape((len(lats), len(lons), d['max_comps']))
    print weights.shape, 'weights shape, check nr timesteps'
    # time_axis = d['time']

    ## Time axis constructed from start and end year, assuming that the last full year is included...
    time_axis = np.linspace(int(d['start_end'][0].split('-')[0]), 
                            int(d['start_end'][1].split('-')[0])+1, len(d['time']))

    ## Number of days in a year (just needed for power spectrum...)
    Fs = 365.

    with PdfPages(save_folder + save_name + '.pdf') as pdf:
        for comp in range(d['max_comps']):
            fig = plt.figure( figsize=(4, 5))
            target_proj = ccrs.PlateCarree(central_longitude=180.0)

            vmax = np.amax(weights[:, :, comp])
            vmin = -vmax    #         print d['explained'].shape
    #         print comp,  d['explained'][comp]
            # title = "No. %d: %.1f%% of variance\n(all %d comps. explain %d%%)" % (comp, 
            #                                     100.*d['explained_var'][comp]/d['total_var'], d['max_comps'], 100*d['explained'])
            title = "No. %d" % (comp)
            
            ax = fig.add_subplot(311, projection=target_proj)
            ## Set up map projection
            ## Potentially adjust this!
            data_proj = ccrs.PlateCarree(central_longitude=0.0)
            ax.coastlines()
            # print weights[:,:,comp]
            cont = ax.contourf(lons, lats, weights[:,:,comp], 
                cmap=plt.get_cmap('RdBu_r'), 
                vmin = vmin,
                vmax = vmax,
                transform=data_proj
                )

            ax.set_title(title, fontsize=10)

            ax.set_global()
            plt.colorbar(cont, ax=ax)

            ax.text(d['comps_max_loc']['x'][comp], d['comps_max_loc']['y'][comp], '%d' % comp, fontsize=12,
                                horizontalalignment='center', verticalalignment='center', color='orange',
                                transform=data_proj)

            # Plot time series
            ax = fig.add_subplot(312)
            ax.set_title("Time series", fontsize=10)
            x = time_axis   #np.arange(len(time_axis))/365. + 50
            series =   d['ts_unmasked'][:,comp] #+ 0*np.sin(2.*np.pi*x/50.) + 0.*np.random.randn(len(x))
            ax.plot(time_axis, series)
 
            # Plot power spectrum
            ax = fig.add_subplot(325)
            ax.set_title("Power spectrum", fontsize=10)
            (psd, freq) = mlab.psd(series, NFFT=len(time_axis), Fs=Fs, scale_by_freq = False, noverlap = None )
            period = 1./freq
            ax.semilogx(period, psd, color = 'black')
            ax.grid(True)
            ax.set_xlabel('Period [years]')
            ax.set_xlim(0.1, 100)

            # Plot autocorrelation
            ax = fig.add_subplot(326)
            ax.set_title("Autocorrelation", fontsize=10)
            ax.acorr(series, maxlags=30)
            ax.grid(True)
            # ax.set_xlabel('lag [days]')
            ax.set_xlabel('lag [3 days]')            
            ax.set_xlim(0, 30)

            # plt.tight_layout()



            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            pdf.savefig()  # saves the current figure into a pdf page



if __name__ == '__main__':


    ##
    ##  Script to generate Varimax loadings and time series
    ##
    ##  All parameters are entered below in the dictionary d. They will
    ##  all be saved together with the results
    ##

    # components=[40,60,80,100]
    components=[100]


    run_script = True
    main_path = './'

    # save_folder = '/media/peer/Firecuda/cmip5_jakob/calc_3dm_varimax/'
    save_folder = main_path  #+'analysis_many_members/varimax/varimax_data/'
    ## Save filename is generated from parameters, see below!
    ### use already calculated weights from the NCAR/NCEP reanalysis
    use_weights_file = True
    weights_filename = 'varimax_SLP_ncar_01-01-1948_31-12-2017_detrend.nc_3dm_comps-%d_months-[6, 7, 8].bin'
    

    plot_loadings = False    # Probably doesn't work on cluster...

    if run_script:
        for i in range(0,len(components)):

            n_comps = components[i]
            weights_filename_here = weights_filename % n_comps


            d = {}
            
            d['data_parameters'] = {
                'folder_name' : './',   
                'load_filename' :  'psl_day_EC-EARTH_rcp85_r1i1p1_19480101-20171231_detrend_9members_cat_regridded.nc',
                # 'load_filename' :  'SLP_ncar_01-01-1948_31-12-2017_detrend.nc',
                # 'varname' : 'slp',
                'varname' : 'psl',
                # 'folder_name' : "/home/jakobrunge/Daten/varimax_components/peer/",
                # 'load_filename' :  "test_xizka_set.nc",
                # 'varname' : 'temp',
                
                'use_cdftime' : True,
                'from_date' : None, #datetime(1848, 1, 1), # datetime(1848, 1, 1),   # datetime object
                'to_date' : None, #datetime(2209, 1, 1), # datetime(1855, 1, 1),
                'anomalize': 'means_variance',
                'anomalize_base' :  None, #(1848, 1855),   # None OR (1948, 2012)
                'slice_lat' : [-89, 89],
                'slice_lon' : None,
                'level' : None,   
                'verbosity' : 2,
            }

            d['preprocessing_parameters'] = {
                'detrend' : False,
                'cos_reweighting' : True,
                'verbosity' : 2,
            }
            
            if use_weights_file:
                # Take varimax_parameters from chosen weights file and add more info
                d['varimax_parameters'] = cPickle.load(open(main_path + weights_filename_here))['varimax_parameters']
                d['varimax_parameters']['use_weights_file'] = use_weights_file
                d['varimax_parameters']['weights_filename'] = weights_filename_here
            else:
                d['varimax_parameters'] = {
                    'use_weights_file' : False,
                    'verbosity' : 2,
                    'truncate_by' : 'max_comps',   # 'max_comps'  OR 'fraction_explained_variance'
                    'fraction_explained_variance' : 0.95,
                    'max_comps' : n_comps,
                    'month_mask' : [6, 7, 8], # [12, 1, 2],    # [1,2,3,4,5,6,7,8,9,10,11,12]
                }

            ## Adapt to your needs:
            save_filename = 'varimax_%s' % (d['data_parameters']['load_filename'])
            save_filename += '_3dm_comps-%s' % (d['varimax_parameters']['max_comps'])
            if use_weights_file:
                save_filename += '_FW'
            if d['varimax_parameters']['month_mask'] is not None:
                save_filename += '_months-%s' % (d['varimax_parameters']['month_mask'])
                save_filename += '.bin'
                
            print("Running script to generate %s" % save_folder + save_filename)
               
            # sys.exit(0)
                
            ##
            ## Start of script
            ##
            geo_object = load_data(**d['data_parameters'])
            
            geo_object = preprocess_data(geo_object, **d['preprocessing_parameters'])
            
            if use_weights_file:
                d['results'] = get_results_from_weights(geo_object, 
                                            weights_filename=main_path + weights_filename_here, 
                                            verbosity=d['varimax_parameters']['verbosity']
                            )           
            else:
                d['results'] = get_varimax_loadings(geo_object, **d['varimax_parameters']
                            )

            cPickle.dump(d, open(save_folder + save_filename, 'wb'))
            
            if plot_loadings:
                generate_pdf(d,  save_folder, save_filename)
                
    else:
        d = cPickle.load(open(plot_load_folder + plot_load_filename))
              
        if plot_loadings:
            generate_pdf(d,  plot_load_folder, plot_load_filename)

    ##
    ## Check get_ts_from_loading
    ##
    # geo_object = load_data(**d['data_parameters'])

    # geo_object = preprocess_data(geo_object, **d['preprocessing_parameters'])

    # ts = get_ts_from_loading(geo_object, 
    #                     weights=d['results']['weights'])

    # assert np.allclose(ts, d['results']['ts_unmasked'])
