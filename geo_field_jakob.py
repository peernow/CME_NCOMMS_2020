"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import math
from datetime import datetime, date
from netCDF4 import Dataset
from netcdftime import utime

class GeoField:
    """
    A class that represents the time series of a geographic field.
    All represented fields have two spatial dimensions (longitude,
    latitude) and one temporal dimension.  Optionally the fields may
    have a height dimension.
    """ 
    
    def __init__(self):
        """
        Initialize to empty data.
        """
        self.d = None
        self.lons = None
        self.lats = None
        self.tm = None
        self.level = None
        self.cos_weights = None
        
    
    def data(self):
        """
        Return a copy of the stored data as a multi-array. If access
        without copying is required for performance reasons, use
        self.d at your own risk.
        """
        return self.d.copy()
        
    
    def use_existing(self, d, lons, lats, tm):
        """
        Initialize with already existing data.
        """
        self.d = d
        self.lons = lons
        self.lats = lats
        self.tm = tm
        
        
    def load(self, netcdf_file, var_name, use_cdftime=False):
        """
        Load GeoData structure from netCDF file.
        """
        d = Dataset(netcdf_file)
        v = d.variables[var_name]
        
        # extract the data
        # self.d = v[:,0,:,:]
        self.d = v[:,:,:]

        self.use_cdftime = use_cdftime
        # print "----> ", self.d.shape
        
        # extract spatial & temporal info
        try:
            self.lons = d.variables['lon'][:]
        except:
            self.lons = d.variables['longitude'][:]

        try:
            self.lats = d.variables['lat'][:]
        except:
            self.lats = d.variables['latitude'][:]

        # self.tm = d.variables['time'][:]
        # self.lons = d.variables['lon'][:]
        # self.lats = d.variables['lat'][:]
        # self.tm = d.variables['time'][:]
        # print d.variables['time'][:]
        # print d.variables['time'].units

        ##
        ## Converting time axis to days since 01-01-01
        ##
        try:
            time_var = d.variables['time']
        except:
            time_var = d.variables['t']
    
        if self.use_cdftime:
            self.cdftime = utime(time_var.units, calendar=time_var.calendar)
            self.tm = time_var[:]
            if 'days' in time_var.units:
                if (self.tm[1] - self.tm[0]) == 1.:
                    self.delta_t  = 'days'
                elif 25 < (self.tm[1] - self.tm[0]) < 40.:
                    self.delta_t  = 'months' 
                else:
                    raise ValueError("Delta time not implemented")
            elif 'hours' in time_var.units:
                if (self.tm[1] - self.tm[0]) == 24.:
                    self.delta_t  = 'days'
                elif 25*24 < (self.tm[1] - self.tm[0]) < 40.*24:
                    self.delta_t  = 'months' 
                else:
                    raise ValueError("Delta time not implemented")
        else:
            if (time_var.units == 'hours since 01-01-01 00:00:0.0'
               or time_var.units == 'hours since 1-1-1 00:00:0.0'):
                self.tm = time_var[:] / 24.0 - 1.0
            elif time_var.units == 'hours since 1800-01-01 00:00:0.0':
                self.tm = (time_var[:] / 24.0 - 1.0) + date(1800, 1, 1).toordinal()
            elif time_var.units == "days since 1850-1-1 00:00:00":
                self.tm = time_var[:] + date(1850, 1, 1).toordinal() 
            elif time_var.units == "days since 1999-09-01 00:00:00":
                self.tm = time_var[:] + date(1999, 9, 1).toordinal() 
            else:
                raise ValueError("d.variables['time'].units style not implemented")
                # self.tm = d.variables['time'][:]


            if (self.tm[1] - self.tm[0]) == 1.:
                self.delta_t  = 'days' #d.variables['time'].delta_t #"0000-01-00 00:00:00"
            elif 25 < (self.tm[1] - self.tm[0]) < 40.:
                self.delta_t  = 'months' 
            else:
                raise ValueError("Delta time not implemented")
            # self.orig_data = np.copy(v)

        if self.use_cdftime:
            self.start_date = self.cdftime.num2date(self.tm[0])
            self.end_date = self.cdftime.num2date(self.tm[-1])
        else:
            self.start_date = date.fromordinal(int(self.tm[0]))
            self.end_date = date.fromordinal(int(self.tm[-1]))

        
        d.close()
        

    def slice_level(self, lev):
        """
        Slice the data and keep only one level (presupposes that the
        data has levels).
        """
        if self.d.ndim > 3:
            self.d = self.d[:, lev, ...]
            # print self.d.shape
        # else:
    

    def slice_date_range(self, date_from, date_to):
        """
        Subselects the date range.  Date_from is inclusive, date_to is
        not.  Modification is in-place due to volume of data.
        """
        
        if self.use_cdftime:
            # print date_from
            # print self.cdftime.num2date(self.cdftime.date2num(datetime(1800, 1, 1)))
            d_start = self.cdftime.date2num(date_from)
            d_stop = self.cdftime.date2num(date_to)
        else:
            d_start = date_from.toordinal()
            d_stop = date_to.toordinal()
        # print date_from, d_start
        # print date_to, d_stop
        # print self.tm

        ndx = np.logical_and(self.tm >= d_start, self.tm < d_stop)
        # print ndx
        self.tm = self.tm[ndx]
        self.d = self.d[ndx, :]
        

    def find_date_ndx(self, dt):
        """
        Return the index that corresponds to the specific day.
        Returns none if the date is not contained in the data.
        """
        if self.use_cdftime:
            d_day = self.cdftime.date2num(dt)
        else:
            d_day = dt.toordinal()
        pos = np.nonzero(self.tm == d_day)
        if len(pos) == 1:
            return pos[0]
        else:
            return None
        
        
    def slice_months(self, months):
        """
        Subselect only certain months, not super efficient but
        useable, since upper bound on len(months) = 12. Modification
        is in-place due to volume of data.
        """
        tm = self.tm
        if self.use_cdftime:
            ndx = filter(lambda i: self.cdftime.num2date(tm[i]).month in months, range(len(tm)))
        else:
            ndx = filter(lambda i: date.fromordinal(int(tm[i])).month in months, range(len(tm)))
        
        # print ndx
        self.tm = tm[ndx]
        self.d = self.d[ndx, ...]
  
    def return_masked_months(self, month_mask):
        """
        Subselect only certain months, not super efficient but
        useable, since upper bound on len(months) = 12. Modification
        is in-place due to volume of data.
        """
        tm = self.tm

        if self.use_cdftime:
            ndx = filter(lambda i: self.cdftime.num2date(tm[i]).month in month_mask, range(len(tm)))
        else:
            ndx = filter(lambda i: date.fromordinal(int(tm[i])).month in month_mask, range(len(tm)))
              
        # print ndx
        # self.tm = tm[ndx]
        time_mask = np.ones(len(self.tm), dtype='bool')
        time_mask[ndx] = False

        return self.d[ndx, ...], time_mask      

    def slice_spatial(self, lons, lats):
        """
        Slice longitude and/or latitude.  None means don't modify
        dimension.  Both arguments are ranges [from, to], both limits
        are inclusive.
        """
        if lons != None:
            lon_ndx = np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= lons[1]))[0]
        else:
            lon_ndx = np.arange(len(self.lons))
            
        if lats != None:
            lat_ndx = np.nonzero(np.logical_and(self.lats >= lats[0], self.lats <= lats[1]))[0]
        else:
            lat_ndx = np.arange(len(self.lats))

        # apply slice to the data (by dimensions, as slicing in two
        # dims at the same time doesn't work)
        d = self.d
        d = d[..., lat_ndx, :]
        self.d = d[..., lon_ndx]
            
        # apply slice to the vars
        self.lons = self.lons[lon_ndx]
        self.lats = self.lats[lat_ndx]
        

    def transform_to_anomalies(self, anomalize_base, anomalize='means_variance'):
        """
        Remove the yearly cycle from the time series.
        """
        # check if data is monthly or daily
        d = self.d
        # delta = self.tm[1] - self.tm[0]

        if anomalize_base is not None:
            if self.use_cdftime:
                ano_start = np.argwhere(self.tm >=  self.cdftime.date2num(datetime(anomalize_base[0], 1, 1))).squeeze()[0]
                ano_end = np.argwhere(self.tm < self.cdftime.date2num(datetime(anomalize_base[1], 1, 1))).squeeze()[-1]
            else:
                ano_start = np.argwhere(self.tm >= date(anomalize_base[0], 1, 1).toordinal()).squeeze()[0]
                ano_end = np.argwhere(self.tm<date(anomalize_base[1], 1, 1).toordinal()).squeeze()[-1]
        else:
            ano_start = 0
            ano_end = len(self.d)

        # print "\tOriginal date range ", date.fromordinal(int(self.tm[0])), date.fromordinal(int(self.tm[-1]))
        # print "\tAno date range ", date.fromordinal(int(self.tm[0])), date.fromordinal(int(self.tm[-1]))

        if self.delta_t == 'days':   # "0000-00-01 00:00:00":
            # daily data
            day, mon = self.extract_day_and_month()

            for mi in range(1, 13):
                month_mask = (mon == mi)
                for di in range(1, 32):
                    sel = np.logical_and(month_mask, day == di)
                    # some days, will be nonexistent, e.g. 30th Feb
                    if np.sum(sel) == 0:
                        continue
                    sel = np.where(sel)[0]
                    # print sel

                    meansel = sel[np.where(((sel>=ano_start) & (sel<=ano_end)))]
                    # sel = sel[np.where(sel)[0]<=ano_end]
                    # print meansel
                    if anomalize == 'means_variance':
                        mn = np.mean(d[meansel, :, :], axis = 0)
                        d[sel, :, : ] -= mn
                        std = np.std(d[meansel, :, :], axis = 0)
                        d[sel, :, : ] /= std
                    elif anomalize == 'means':
                        mn = np.mean(d[meansel, :, :], axis = 0)
                        d[sel, :, : ] -= mn

        # elif abs(delta - 30) < 3.0:
        elif self.delta_t == 'months': # "0000-01-00 00:00:00":
            # monthly data
            for i in range(12):
                # print i, np.arange(ano_start + i, ano_end + i, 12)
                if anomalize == 'means_variance':
                    mn = np.mean(d[ano_start + i:ano_end + i:12, :, :], axis = 0)
                    d[i::12, :, :] -= mn
                    std = np.std(d[ano_start + i:ano_end + i:12, :, :], axis = 0)
                    d[i::12, :, :] /= std
                elif anomalize == 'means':
                    mn = np.mean(d[ano_start + i:ano_end + i:12, :, :], axis = 0)
                    d[i::12, :, :] -= mn
        else:
            raise ValueError("Unknown temporal sampling in geographical field: %s" % self.delta_t)
      

        # if self.delta_t == 'days':
        #     for i in range(365):
        #         assert np.amax(d[i::365].mean(axis=0)) < 1E-2, np.amax(d[::365].mean(axis=0))  
        # elif self.delta_t == 'months':
        #     for i in range(12):
        #         assert np.amax(d[i::12].mean(axis=0)) < 1E-2, np.amax(d[::12].mean(axis=0))

    # def normalize_variance(self, anomalize_base):
    #     """
    #     Normalize the variance of monthly or daily values.  A calendar
    #     strategy is applied for daily data.
    #     """

    #     if anomalize_base is not None:
    #         ano_start = np.argwhere(self.tm==date(anomalize_base[0], 1, 1).toordinal()).squeeze()
    #         ano_end = np.argwhere(self.tm==date(anomalize_base[1], 1, 1).toordinal()).squeeze()
    #     else:
    #         ano_start = 0
    #         ano_end = len(self.d)

    #     # print ano_start, ano_end

    #     # check if data is monthly
    #     d = self.d
    #     delta = self.tm[1] - self.tm[0]

    #     if self.delta_t == 'days': # "0000-00-01 00:00:00":
    #         # daily data
    #         day, mon = self.extract_day_and_month()

    #         for mi in range(1, 13):
    #             month_mask = (mon == mi)
    #             for di in range(1, 32):
    #                 sel = np.logical_and(month_mask, day == di)
    #                 # some days, will be nonexistent, e.g. 30th Feb
    #                 if np.sum(sel) == 0:
    #                     continue
    #                 std = np.std(d[sel, :, :], axis = 0)
    #                 if np.any(std == 0.0):
    #                     print("WARN: some zero stdevs found for date %d.%d."
    #                           % (di, mi))
    #                     std[std == 0.0] = 1.0
    #                 d[sel, :, : ] /= std

    #     elif self.delta_t == 'months': #"0000-01-00 00:00:00":
    #         # monthly data
    #         for i in range(12):
    #             mn = np.std(d[ano_start + i:ano_end + i:12, :, :], axis = 0)
    #             d[i::12, :, :] /= mn
    #     else:
    #         raise ValueError("Unknown temporal sampling in geographical field: %s" % self.delta_t)


    def sample_temporal_bootstrap(self):
        """
        Return a temporal bootstrap sample.
        """
        # select samples at random
        num_tm = len(self.tm)
        ndx = sp.random.random_integers(0, num_tm - 1, size = (num_tm,))
        
        # return a copy of the resampled dataset
        return self.d[ndx, ...].copy(), ndx
    
    
    def spatial_dims(self):
        """
        Return the spatial dimensions of the data.  Spatial dimensions are all dimensions
        after the first one, which corresponds to time.
        """
        return self.d.shape[1:]
    
    
    def reshape_flat_field(self, f):
        """
        Reshape a field that has been spatially flattened back into shape that
        corresponds to the spatial dimensions of this field.  It is assumed that
        f is a 2D array with axis 0 spatial and axis 1 temporal.  The matrix will
        be transposed and the spatial dimension reshaped back to match the spatial
        dimensions of the data in this field.
        """
        n = f.shape[1]
        f = f.transpose()
        new_shape = [n] + list(self.spatial_dims())
        return f.reshape(new_shape)


    def detrend(self):
        """
        Apply scipy.signal.detrend to axis 0 (time) of the data.
        """
        self.d = sps.detrend(self.d, axis = 0)
    
    
    def qea_latitude_weights(self):
        """
        Return a grid which contains the scaling factors to rescale each time series
        by sqrt(cos(lattitude)).
        """
        # if cached, return cached version
        if self.cos_weights != None:
            return self.cos_weights
        
        # if not cached recompute, cache and return
        cos_weights = np.zeros_like(self.d[0, :, :])
        lats = self.lats
        ##### why is this squre rooted????? (**0.5), somewhere else taken square I assume-> cov?
        for lat_ndx in range(len(lats)):
                cos_weights[lat_ndx, :] = math.cos(lats[lat_ndx] * math.pi / 180) ** 0.5
                 
        self.cos_weights = cos_weights
        self.d *= cos_weights


    def apply_filter(self, b, a):
        """
        Apply a filter in b, a form (uses filtfilt) to each time series.
        """
        d = self.d
        
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                d[:, i,j] = sps.filtfilt(b, a, d[:, i, j])


    def extract_day_and_month(self):
        """
        Convert the self.tm data into two arrays -- day of month
        and month of year.
        """
        Ndays = len(self.tm)
        days = np.zeros((Ndays,))
        months = np.zeros((Ndays,))

        for i, d in zip(range(Ndays), self.tm):
            if self.use_cdftime:
                dt = self.cdftime.num2date(d)
            else:
                dt = date.fromordinal(int(d))
            days[i] = dt.day
            months[i] = dt.month
        
        return days, months

if __name__ == '__main__':
    pass
