import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import scipy.interpolate as si
import dask.array as da
from dask import delayed
import os
import shutil
import struct
import copy
import matplotlib.pyplot as pl

from xml.etree import ElementTree
from collections import OrderedDict

def _to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def _read_binary_grid(f):
    z = struct.unpack('q', f.read(8))[0]
    y = struct.unpack('q', f.read(8))[0]
    x = struct.unpack('q', f.read(8))[0]
    n = x * y * z

    xx = np.fromfile(f, dtype=np.float32, count=n)
    yy = np.fromfile(f, dtype=np.float32, count=n)
    zz = np.fromfile(f, dtype=np.float32, count=n)

    xx = np.transpose(xx.reshape([z, y, x]))
    yy = np.transpose(yy.reshape([z, y, x]))
    zz = np.transpose(zz.reshape([z, y, x]))
    return x, y, z, xx, yy, zz


class Result(object):

    def __init__(self, xml_file='', reprocess=False, result_root=None, data_root=None, piv_pass=-1, from_nc=False, norm_dims=False):
        self._xml_file = xml_file
        self._ds = None
        self._reprocess = reprocess
        self._from_nc = from_nc
        self._norm_dims = norm_dims

        self._xml_data = ElementTree.parse(xml_file)
        self._xml_root = self._xml_data.getroot()
        self.piv_pass = piv_pass

        # Parse the XML config file to identify the root folder that data processed in
        if result_root is None:
            result_root = self._xml_root.findall('runFolder')[0].text

        if result_root[-1] != '/':
            result_root = result_root + '/'
        self._result_root = result_root

        self.run_id = self._result_root.split('/')[-2]
        self.piv_tag = self._xml_root.findall('piv/outputTag')[0].text

        if (self.piv_pass >= 2) or (self.piv_pass == -1):
            grid = 2
        else:
            grid = self.piv_pass
        self.grid = grid

        # Determine the root folder for the compiled netCDF, if None use result_root
        if data_root is None:
            self._data_root = self._result_root
        else:
            # take the last folder as the
            if data_root[-1] != '/':
                data_root = data_root + '/'
            self._data_root = data_root + self.run_id
            if not os.path.isdir(self._data_root):
                os.mkdir(self._data_root)

        # Implement naming convention for netCDF file
        #self._result_file = self._data_root + '/' + self.run_id + '_' + self.piv_tag + '_velocity.nc'
        self._result_file = self._data_root + '/' + self.run_id + '_' + self.piv_tag + '_velocity.zarr'

    @property
    def result_file(self):
        return self._result_file

    def _loadgrid(self):

        try:
            with open(self._result_root + "xyz_grid_" + self.piv_tag + "_pass%i.bin" % self.grid, "rb") as f:
                x, y, z, xx, yy, zz = _read_binary_grid(f)
        except FileNotFoundError:
            print('Unable to locate grid for ' + self._xml_file)
            return None, None, None

        # Unit conversions here should probably have been done in step3_piv.cpp
        # The spatial grid is output in pixels - convert to meters
        px = float(self._xml_root.findall('capture/dx')[0].text)  # mm/px
        py = float(self._xml_root.findall('capture/dy')[0].text)  # mm/px
        pz = float(self._xml_root.findall('reprojection/dz')[0].text)  # mm/px

        xx = xx[:, 0, 0] * px / 1000.
        yy = yy[0, :, 0] * py / 1000.
        zz = zz[0, 0, :] * pz / 1000.

        return xx, yy, zz

    @delayed
    def _loadtimestep(fn):
        with open(fn, "rb") as f:
            x, y, z, u, v, w = _read_binary_grid(f)

            # this should be subsequent filter rather than written to 'raw' velocity file
            sigma = 0.5
            u = nd.filters.gaussian_filter(u, sigma)
            v = nd.filters.gaussian_filter(v, sigma)
            w = nd.filters.gaussian_filter(w, sigma)

        return np.stack([u,v,w],axis=-1)
        #return u, v, w 

    def _loaddata(self, xx, yy, zz):

        frames = list()
        all_files = os.listdir(self._result_root + 'velocity/')
        for fn in sorted(all_files):
            if fn.startswith(self.piv_tag):
                frames.append(fn)

#        tt = np.zeros((len(frames),)).astype(pd.tslib.Timestamp)
#        ut = da.zeros((len(xx), len(yy), len(zz), len(frames)))
#        vt = da.zeros((len(xx), len(yy), len(zz), len(frames)))
#        wt = da.zeros((len(xx), len(yy), len(zz), len(frames)))
        if len(frames) > 0:
            tt = np.array([pd.to_datetime(fn.split('_')[-1][:-4], format='%Y-%m-%d-%H-%M-%S.%f') for fn in frames])

            lazydata = [self._loadtimestep(self._result_root + "velocity/" + fn) for fn in frames]
            arrays = [da.from_delayed(lazy_uvw,dtype=np.float32,shape=(len(xx), len(yy), len(zz), 3)) for lazy_uvw in lazydata]
            uvw = da.stack(arrays,axis=0)/1000.
            uvw = uvw.rechunk((50,len(xx), len(yy), len(zz), 3))
        else:
            tt = None
            uvw = None
#        for it, fn in enumerate(frames):
#            try:
#               # print('%.0f - %s' % (it, fn))
#                uvw.append(da.from_delayed(_loadtimestep(fn)
#                u,v,w=_loadtimestep(fn)
#                with open(self._result_root + "velocity/" + fn, "rb") as f:
#                    x, y, z, u, v, w = _read_binary_grid(f)
#
#                    # this should be subsequent filter rather than written to 'raw' velocity file
#                    sigma = 0.5
#                    u = nd.filters.gaussian_filter(u, sigma)
#                    v = nd.filters.gaussian_filter(v, sigma)
#                    w = nd.filters.gaussian_filter(w, sigma)
#
#                    # convert to meters/s
#                ut[:, :, :, it] = u / 1000.
#                vt[:, :, :, it] = v / 1000.
#                wt[:, :, :, it] = w / 1000.
#
#                tt[it] = pd.to_datetime(fn.split('_')[-1][:-4], format='%Y-%m-%d-%H-%M-%S.%f')
#
#            except IOError:
#                print('Cannot open ' + fn)
#                shp = (len(xx), len(yy), len(zz))
#                ut[:, :, :, it] = np.ones(shp) * np.nan
#                vt[:, :, :, it] = np.ones(shp) * np.nan
#                wt[:, :, :, it] = np.ones(shp) * np.nan

        return tt, uvw #, vt, wt

    def _append_CF_attrs(self, ds):
        ds.attrs['title'] = 'Three dimensional particle imaging velocimetry data'
        ds.attrs['institution'] = 'University of Western Australia'
        ds.attrs['source'] = '3DPIV applied to Synthetic Aperture Imagery'
        ds.attrs['references'] = 'Branson, P. M., Shallow island wake stability and upwelling in tidal ﬂow measured by 3D particle imaging velocimetry, Ph.D. thesis, Ocean Graduate School, University of Western Australia, 2018.'

    def _append_attrs(self, ds):
        pdx = float(self._xml_root.findall('capture/dx')[0].text)
        pdy = float(self._xml_root.findall('capture/dx')[0].text)
        pdz = float(self._xml_root.findall('reprojection/dz')[0].text)
        burst_num = int(self._xml_root.findall('capture/nBurst')[0].text)
        burst_interval = float(self._xml_root.findall('capture/burstInterval')[0].text)
        capture_dt = float(self._xml_root.findall('capture/dt')[0].text)
        step_ensemble = float(self._xml_root.findall('piv/stepEnsemble')[0].text)
        piv_step_frame = float(self._xml_root.findall('piv/stepFrame')[0].text)
        piv_n_ensemble = float(self._xml_root.findall('piv/nEnsemble')[0].text)

        ds.attrs['run_id'] = self.run_id
        ds.attrs['dx'] = float(ds.x[1] - ds.x[0])
        ds.attrs['dy'] = float(ds.y[1] - ds.y[0])
        ds.attrs['dz'] = float(ds.z[1] - ds.z[0])
        ds.attrs['pdx'] = pdx / 1000.
        ds.attrs['pdy'] = pdy / 1000.
        ds.attrs['pdz'] = pdz / 1000.
        ds.attrs['captureDt'] = capture_dt
        ds.attrs['capture_burst_interval'] = burst_interval
        ds.attrs['capture_n_burst'] = burst_num
        ds.attrs['piv_step_ensemble'] = step_ensemble
        ds.attrs['piv_step_frame'] = piv_step_frame
        ds.attrs['piv_n_ensemble'] = piv_n_ensemble
        ds.attrs['piv_grid'] = self.grid
        ds.attrs['piv_tag'] = self.piv_tag

        if self._xml_root.find('flow') is not None:
            for ele in self._xml_root.find('flow'):
                ds.attrs[ele.tag] = _to_num(ele.text)

        with open(self._xml_file) as f:
            xml_string = f.read()

        ds.attrs['XML'] = xml_string

    @property
    def ds(self):
        if self._ds is None:
            file_exists = os.path.exists(self._result_file)

            reprocess = not file_exists or self._reprocess

            if reprocess:
                if file_exists:
                    print('Old file exists ' + self._result_file)
                    #print('Removing old file ' + self._result_file)
                    #shutil.rmtree(self._result_file)

                ds_data = OrderedDict()

                to_seconds = np.vectorize(lambda x: x.seconds + x.microseconds / 1E6)

                print('Processing binary data...')
                xx, yy, zz = self._loadgrid()
                if xx is None:
                    if self._from_nc:
                        print('Processing existing netcdf...')
                        fn = self._result_file[:-5] + '_QC_raw.nc'
                        if os.path.exists(fn):
                            ds_temp = xr.open_dataset(self._result_file[:-5] + '_QC_raw.nc',chunks={'time':50})
                            u = da.transpose(ds_temp['U'].data,axes=[3,0,1,2])
                            v = da.transpose(ds_temp['V'].data,axes=[3,0,1,2])
                            w = da.transpose(ds_temp['W'].data,axes=[3,0,1,2])
                            tt = ds_temp['time']
                            te = (tt - tt[0]) / np.timedelta64(1, 's')
                            xx = ds_temp['x'].values
                            yy = ds_temp['y'].values
                            zz = ds_temp['z'].values
                        else:
                            print('USING OLD ZARR DATA')
                            ds_temp = xr.open_zarr(self._result_file)
                            u = da.transpose(ds_temp['U'].data,axes=[3,0,1,2])
                            v = da.transpose(ds_temp['V'].data,axes=[3,0,1,2])
                            w = da.transpose(ds_temp['W'].data,axes=[3,0,1,2])
                            tt = ds_temp['time']
                            te = (tt - tt[0]) / np.timedelta64(1, 's')
                            xx = ds_temp['x'].values
                            yy = ds_temp['y'].values
                            zz = ds_temp['z'].values
                            print('ERROR: No NetCDF data found for ' + self._xml_file)
                            #return None
                            # print(u.shape)
                 
                else:
                    tt, uvw = self._loaddata(xx, yy, zz)
                    if tt is None:
                        print('ERROR: No binary data found for ' + self._xml_file)
                        return None

                    # calculate the elapsed time from the Timestamp objects and then convert to datetime64 datatype
                    te = to_seconds(tt - tt[0])
                    tt = pd.to_datetime(tt)
                    uvw = uvw.persist()
                    u = uvw[:,:,:,:,0]
                    v = uvw[:,:,:,:,1]
                    w = uvw[:,:,:,:,2]

#                    u = xr.DataArray(uvw[:,:,:,:,0], coords=[tt, xx, yy, zz], dims=['time','x', 'y', 'z'],
#                                     name='U', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
#                    v = xr.DataArray(uvw[:,:,:,:,1], coords=[tt, xx, yy, zz], dims=['time', 'x', 'y', 'z'],
#                                     name='V', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
#                    w = xr.DataArray(uvw[:,:,:,:,2], coords=[tt, xx, yy, zz], dims=['time', 'x', 'y', 'z'],
#                                     name='W', attrs={'standard_name': 'upward_sea_water_velocity', 'units': 'm s-1'})

                if xx is None:
                    print('No data found')
                    return None

                u = u.persist()
                v = v.persist()
                w = w.persist()

                dx = float(xx[1] - xx[0])
                dy = float(yy[1] - yy[0])
                dz = float(zz[1] - zz[0])

                
                if self._norm_dims:
                    exp = self._result_root.split('/')[4]
                    runSheet = pd.read_csv('~/RunSheet-%s.csv' % exp)
                    runSheet = runSheet.set_index('RunID')
                    runDetails = runSheet.ix[int(self.run_id[-2:])]

                    T = runDetails['T (s)']
                    h = runDetails['h (m)']
                    D = runDetails['D (m)']
            
                    ww = te / T
                    om = 2. * np.pi / T
                    d_s = (2. * 1E-6 / om) ** 0.5
                    bl = 3. * np.pi / 4. * d_s
                    
                    if exp == 'Exp6':
                        if D == 0.1:
                            dy_c = (188. + 82.)/2
                            dx_c = 39.25
                            cx = dx_c / 1000.
                            cy = dy_c / 1000.
                        else:
                            dy_c = (806. + 287.) / 2. * 0.22
                            dx_c = 113 * 0.22
                            cx = dx_c / 1000.
                            cy = dy_c / 1000.   
                    elif exp == 'Exp8':
                        dy_c = 624 * 0.22
                        dx_c = 15
                        cx = dx_c / 1000.
                        cy = dy_c / 1000.
                    xn = (xx + (D / 2. - cx)) / D
                    yn = (yy - cy) / D
                    zn = zz / h
 
                    xnm, ynm = np.meshgrid(xn, yn)
                    rr = np.sqrt(xnm ** 2. + ynm ** 2)
                    cylMask = rr < 0.5

                    nanPlane = np.ones(cylMask.shape)
                    nanPlane[cylMask] = np.nan
                    nanPlane = nanPlane.T
                    nanPlane = nanPlane[np.newaxis,:, :, np.newaxis]

                    u = u * nanPlane
                    v = v * nanPlane
                    w = w * nanPlane

                    if D == 0.1:
                        xInds = xn > 3.
                    else:
                        xInds = xn > 2.

                    blInd = np.argmax(zn > bl / h)
                    blPlane = int(round(blInd))

                    Ue = u[:,xInds,:,:]
                    Ue_bar = da.nanmean(Ue,axis=(1,2,3)).compute()
                    Ue_bl = da.nanmean(Ue[:,:,:,blPlane],axis=(1,2)).compute()

                    inds = ~np.isnan(Ue_bl)

                    xv = ww[inds] % 1.
                    xv = xv + np.random.normal(scale=1E-6,size=xv.shape)
                    yv = Ue_bl[inds]
                    xy = np.stack([np.concatenate([xv-1.,xv,xv+1.]),np.concatenate([yv,yv,yv])]).T
                    xy = xy[xy[:,0].argsort(),:]
                    xi = np.linspace(-0.5,1.5,len(xv)/8)
                    n = np.nanmax(xy[:,1])
                    # print(n)
                    # fig,ax = pl.subplots()
                    # ax.scatter(xy[:,0],xy[:,1]/n)
                    # print(xy)
                    spl = si.LSQUnivariateSpline(xy[:,0],xy[:,1]/n,t=xi,k=3)
                    roots = spl.roots()
                    der = spl.derivative()
                    slope=der(roots)
                    inds = np.min(np.where(slope>0))
                    dt=(roots[inds]%1.).mean()-0.5

                    tpx=np.arange(0,0.5,0.001)
                    U0_bl=np.abs(spl(tpx+dt).min()*n)
                    ws = ww - dt
                    Ue_spl = spl((ws-0.5)%1.0+dt)*n*-1.0

                    #maxima = spl.derivative().roots()
                    #Umax = spl(maxima)
                    #UminIdx = np.argmin(Umax)
                    #U0_bl = np.abs(Umax[UminIdx]*n)

                    #ww_at_min = maxima[UminIdx]
                    #ws = ww - ww_at_min + 0.25

                    inds = ~np.isnan(Ue_bar)

                    xv = ww[inds] % 1.
                    xv = xv + np.random.normal(scale=1E-6,size=xv.shape)
                    yv = Ue_bar[inds]
                    xy = np.stack([np.concatenate([xv-1.,xv,xv+1.]),np.concatenate([yv,yv,yv])]).T
                    xy = xy[xy[:,0].argsort(),:]
                    xi = np.linspace(-0.5,1.5,len(xv)/8)
                    n = np.nanmax(xy[:,1])
                    spl = si.LSQUnivariateSpline(xy[:,0],xy[:,1]/n,t=xi,k=4)
                    maxima = spl.derivative().roots()
                    Umax = spl(maxima)
                    UminIdx = np.argmin(Umax)
                    U0_bar = np.abs(Umax[UminIdx]*n)

                    ww = xr.DataArray(ww, coords=[tt, ], dims=['time', ])
                    ws = xr.DataArray(ws-0.5, coords=[tt, ], dims=['time', ])

                    xn = xr.DataArray(xn, coords=[xx, ], dims=['x', ])
                    yn = xr.DataArray(yn, coords=[yy, ], dims=['y', ])
                    zn = xr.DataArray(zn, coords=[zz, ], dims=['z', ])
                    
                    Ue_bar = xr.DataArray(Ue_bar, coords=[tt, ], dims=['time', ])
                    Ue_bl = xr.DataArray(Ue_bl, coords=[tt, ], dims=['time', ])
                    Ue_spl = xr.DataArray(Ue_spl, coords=[tt, ], dims=['time', ])

                    ds_data['ww'] = ww
                    ds_data['ws'] = ws

                    ds_data['xn'] = xn
                    ds_data['yn'] = yn
                    ds_data['zn'] = zn

                    ds_data['Ue_bar'] = Ue_bar
                    ds_data['Ue_bl'] = Ue_bl
                    ds_data['Ue_spl'] = Ue_spl


                te = xr.DataArray(te, coords=[tt, ], dims=['time', ])

                dims = ['time','x', 'y', 'z']
                coords = [tt, xx, yy, zz]

                ds_data['U'] = xr.DataArray(u, coords=coords, dims=dims,
                                     name='U', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
                ds_data['V'] = xr.DataArray(v, coords=coords, dims=dims,
                                     name='V', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
                ds_data['W'] = xr.DataArray(w, coords=coords, dims=dims,
                                     name='W', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
                ds_data['te'] = te


                # stdV = da.nanstd(v)
                # stdW = da.nanstd(w)
                # thres=7.
                if 'U0_bl' in locals():
                    condition = (da.fabs(v)/U0_bl>1.5) | (da.fabs(w)/U0_bl>0.6)
                    for var in ['U','V','W']:
                        ds_data[var].data=da.where(condition,np.nan,ds_data[var].data)
            
                piv_step_frame = float(self._xml_root.findall('piv/stepFrame')[0].text)

                print('Calculating tensor')
                # j = jacobianConv(ds.U, ds.V, ds.W, dx, dy, dz, sigma=1.5)
                j = jacobianDask(u,v,w, piv_step_frame, dx, dy, dz)
                print('Done') 
                #j = da.from_array(j,chunks=(20,-1,-1,-1,-1,-1))

#                j = jacobianDask(uvw[:,:,:,:,0],uvw[:,:,:,:,1], uvw[:,:,:,:,2], piv_step_frame, dx, dy, dz)
                jT = da.transpose(j,axes=[0,1,2,3,5,4])


#                j = j.persist()
#                jT = jT.persist()
        
                jacobianNorm = da.sqrt(da.nansum(da.nansum(j**2.,axis=-1),axis=-1))
    
                strainTensor = (j + jT) / 2.
                vorticityTensor = (j - jT) / 2.

                strainTensorNorm = da.sqrt(da.nansum(da.nansum(strainTensor ** 2.,axis=-1),axis=-1))
                vorticityTensorNorm = da.sqrt(da.nansum(da.nansum(vorticityTensor ** 2.,axis=-1),axis=-1))
                divergence = j[:,:,:,:,0,0] + j[:,:,:,:,1,1] + j[:,:,:,:,2,2] 
                # print(divergence) 
                omx = vorticityTensor[:, :, :, :, 2, 1] * 2.
                omy = vorticityTensor[:, :, :, :, 0, 2] * 2.
                omz = vorticityTensor[:, :, :, :, 1, 0] * 2.
    
                divNorm = divergence / jacobianNorm

#                divNorm = divNorm.persist()

#                divNorm_mean = da.nanmean(divNorm)
#                divNorm_std = da.nanstd(divNorm)
    
                dims = ['x','y','z']
                comp = ['u','v','w']
    
                ds_data['jacobian'] = xr.DataArray(j,coords=[tt, xx, yy, zz, comp, dims],
                                                     dims=['time','x', 'y', 'z','comp','dims'],
                                                     name='jacobian')
    
                ds_data['jacobianNorm'] = xr.DataArray(jacobianNorm,
                                                       coords=[tt, xx, yy, zz],
                                                       dims=['time','x', 'y', 'z'],
                                                       name='jacobianNorm')
              
                ds_data['strainTensor'] = xr.DataArray(strainTensor,
                                                       coords=[tt, xx, yy, zz, comp, dims],
                                                       dims=['time','x', 'y', 'z', 'comp', 'dims'],
                                                       name='strainTensor')
    
                ds_data['vorticityTensor'] = xr.DataArray(vorticityTensor,
                                                       coords=[tt, xx, yy, zz, comp, dims],
                                                       dims=['time','x', 'y', 'z', 'comp', 'dims'],
                                                       name='vorticityTensor')
    
                ds_data['vorticityNorm'] = xr.DataArray(vorticityTensorNorm,
                                                       coords=[tt, xx, yy, zz],
                                                       dims=['time','x', 'y', 'z'],
                                                       name='vorticityNorm')
    
                ds_data['strainNorm'] = xr.DataArray(strainTensorNorm,
                                                       coords=[tt, xx, yy, zz],
                                                       dims=['time','x', 'y', 'z'],
                                                       name='strainNorm')
    
                ds_data['divergence'] = xr.DataArray(divergence,
                                                       coords=[tt, xx, yy, zz],
                                                       dims=['time','x', 'y', 'z'],
                                                       name='divergence')

                ds_data['omx'] = xr.DataArray(omx,
                                              coords=[tt, xx, yy, zz],
                                              dims=['time','x', 'y', 'z'],
                                              name='omx')

                ds_data['omy'] = xr.DataArray(omy,
                                              coords=[tt, xx, yy, zz],
                                              dims=['time','x', 'y', 'z'],
                                              name='omy')

                ds_data['omz'] = xr.DataArray(omz,
                                              coords=[tt, xx, yy, zz],
                                              dims=['time','x', 'y', 'z'],
                                              name='omz')

                ds_data['divNorm'] = xr.DataArray(divNorm,
                                              coords=[tt, xx, yy, zz],
                                              dims=['time','x', 'y', 'z'],
                                              name='divNorm')

#                ds_data['divNorm_mean'] = xr.DataArray(divNorm_mean)
#                ds_data['divNorm_std'] = xr.DataArray(divNorm_std)

                ds = xr.Dataset(ds_data)
#                if self._from_nc:
#                    for k,v in ds_temp.attrs.items():
#                        ds.attrs[k]=v
                #ds = ds.chunk({'time': 20})

                self._append_CF_attrs(ds)
                self._append_attrs(ds)
                ds.attrs['filename'] = self._result_file

                if self._norm_dims:

                    KC = U0_bl*T/D
                    delta = (2.*np.pi*d_s)/h
                    S = delta/KC

                    ds.attrs['T'] = T
                    ds.attrs['h'] = h
                    ds.attrs['D'] = D
                    ds.attrs['U0_bl'] = U0_bl
                    ds.attrs['U0_bar'] = U0_bar
                    ds.attrs['KC'] = KC
                    ds.attrs['S'] = S
                    ds.attrs['Delta+'] = ((1E-6*T)**0.5)/h
                    ds.attrs['Delta_l'] = 2*np.pi*d_s
                    ds.attrs['Delta_s'] = d_s
                    ds.attrs['Re_D'] = U0_bl*D/1E-6
                    ds.attrs['Beta'] = D**2./(1E-6*T)

                delta = (ds.attrs['dx'] * ds.attrs['dy'] * ds.attrs['dz']) ** (1./3.)
                dpx = (ds.attrs['pdx'] * ds.attrs['pdy'] * ds.attrs['pdz']) ** (1./3.)
                delta_px = delta / dpx
                dt = ds.attrs['piv_step_ensemble']

#                divRMS = da.sqrt(da.nanmean((divergence * dt) ** 2.))
#                divRMS = divRMS.persist()
#                vorticityTensorNorm.persist()
#                velocityError = divRMS/((3./(2.*delta_px**2.))**0.5)
               # print(da.percentile(ds_new['vorticityTensorNorm'].data.ravel(),99.))
               # print(ds_new['divRMS'])
               # print(ds_new['divNorm_mean'])
#                vorticityError = divRMS/dt/da.percentile(vorticityTensorNorm.ravel(),99.)

#                divNorm_mean = da.nanmean(divNorm)
#                divNorm_std = da.nanstd(divNorm)

                # print("initial save")
                #ds.to_zarr(self._result_file,compute=False)
                #ds = xr.open_zarr(self._result_file)

#                xstart = np.argmax(xx > 0.05)
#                ystart = np.argmax(yy > 0.07)

                divRMS = da.sqrt(da.nanmean((divergence * dt) ** 2.))#.compute()
                #divNorm = divergence / jacobianNorm
                #divNorm = divNorm.compute()
                #divNorm_mean = da.nanmean(divNorm).compute()
                #divNorm_std = da.nanstd(divNorm).compute()
                velocityError = divRMS/((3./(2.*delta_px**2.))**0.5)
                vortNorm = vorticityTensorNorm#.compute()
                
                vorticityError = divRMS/dt/np.percentile(vortNorm.ravel(),99.)               

                velocityError, vorticityError = da.compute(velocityError, vorticityError)
            
                #ds.attrs['divNorm_mean'] = divNorm_mean
                #ds.attrs['divNorm_std'] = divNorm_std
                ds.attrs['velocityError'] = velocityError
                ds.attrs['vorticityError'] = vorticityError

                if self._norm_dims:
                    xInds = (xn>0.5) & (xn<2.65)
                    yInds = (yn > -0.75) & (yn < 0.75)
                else:
                    xInds = range(len(ds['x']))
                    yInds = range(len(ds['y']))
                vrms = (ds['V'][:,xInds,yInds,:] ** 2.).mean(dim=['time','x', 'y', 'z']) ** 0.5
                wrms = (ds['W'][:,xInds,yInds,:] ** 2.).mean(dim=['time','x', 'y', 'z']) ** 0.5
                ds.attrs['Vrms'] = float(vrms.compute())
                ds.attrs['Wrms'] = float(wrms.compute())

                #fig,ax = pl.subplots()
                #ax.plot(ds.ws,ds.Ue_spl/U0_bl,color='k')
                #ax.plot(ds.ws,ds.Ue_bl/U0_bl,color='g')
                #ax.set_xlabel(r'$t/T$')
                #ax.set_ylabel(r'$U_{bl}/U_0$')
                #fig.savefig(self._result_file[:-4] + 'png',dpi=125)
                #pl.close(fig)
                # print("second save")
                #ds.to_netcdf(self._result_file)
                ds.to_zarr(self._result_file,mode='w')
                
                print('Cached ' + self._result_file)

                #ds = xr.open_dataset(self._result_file,chunks={'time':20})
                ds = xr.open_zarr(self._result_file)
                ds.attrs['filename'] = self._result_file
            else:
                #ds = xr.open_dataset(self._result_file,chunks={'time':20})
                ds = xr.open_zarr(self._result_file)
                ds.attrs['filename'] = self._result_file

            self._ds = ds

        return self._ds


class DerivedDataset(object):

    def __init__(self, xr_obj, **kwargs):
        self._parent = xr_obj
        self.cache = None

    def __call__(self, tag="DDTAG", reprocess=False):
        self._tag = tag
        self._reprocess = reprocess
        if self.cache is None:
            file_exists = os.path.exists(self.cache_file)
            reprocess = not file_exists or self._reprocess
            if not reprocess:
                #_cache_ds = xr.open_dataset(self.cache_file,chunks={'time':20})
                _cache_ds = xr.open_zarr(self.cache_file)
                self.cache = _cache_ds
            if file_exists and reprocess:
                print('Removing old file ' + self.cache_file)
                shutil.rmtree(self.cache_file)


    @property
    def parent(self):
        return self._parent

    @property
    def cache_file(self):
        fn = self.parent.attrs['filename']
        fn = fn[:fn.rfind('.')] + '.' + self._tag + '.zarr'
        return fn

    def to_file(self):
        self.cache.attrs['filename'] = self.cache_file
        #self.cache.to_netcdf(self.cache_file)
        print("to_zarr")
        self.cache.to_zarr(self.cache_file)

        #self.cache = xr.open_dataset(self.cache_file,chunks={'time':20})
        self.cache = xr.open_zarr(self.cache_file)


# from .accessor import register_result_accessor


def jacobianDask(u, v, w, dt, dx, dy, dz):

    du = da.gradient(u, dx, dy, dz, dt, axis=(1, 2, 3))
    dv = da.gradient(v, dx, dy, dz, dt, axis=(1, 2, 3))
    dw = da.gradient(w, dx, dy, dz, dt, axis=(1, 2, 3))

    J = da.stack((da.stack(du, axis=-1), da.stack(dv, axis=-1), da.stack(dw, axis=-1)), axis=-2)

    return J

def jacobian(u, v, w, dt, dx, dy, dz):

    du = np.gradient(u, dx, dy, dz, axis=(1, 2, 3))
    dv = np.gradient(v, dx, dy, dz, axis=(1, 2, 3))
    dw = np.gradient(w, dx, dy, dz, axis=(1, 2, 3))

    J = np.stack((np.stack(du, axis=-1), np.stack(dv, axis=-1), np.stack(dw, axis=-1)), axis=-2)

    return J


@xr.register_dataset_accessor('stats')
class Stats(DerivedDataset):

    # def __init__(self, xr_obj):
    #     super(Stats, self).__init__(xr_obj)

    def __call__(self, tag='SDTAG', reprocess=False):
        super().__call__(tag='stats-' + tag, reprocess=reprocess)

        if self.cache is None:
            ds = self.parent
            ds_new = xr.Dataset()
            ds_new.attrs = ds.attrs

            rho = 1035.

            dx = ds.dx
            dy = ds.dy
            dz = ds.dz

            print('Calculating stats')
            for v in ['U', 'V', 'W']:
                ds2 = ds[v] ** 2.
                ds_new[v + 't_rms'] = ds2.mean(dim=['x', 'y', 'z']) ** 0.5
                ds_new[v + 't_mean'] = ds[v].mean(dim=['x', 'y', 'z'])
                ds_new.attrs[v + '_mean'] = float(ds[v].mean(dim=['x', 'y', 'z', 'time']))
                ds_new.attrs[v + '_rms'] = float(ds2.mean(dim=['x', 'y', 'z', 'time']) ** 0.5)

            print('Calculating kinetic energy')
            Eht = 0.5 * rho * dx * dy * dz * (ds.U ** 2. + ds.V ** 2.).sum(dim=['x', 'y', 'z'])
            Eut = 0.5 * rho * dx * dy * dz * (ds.U ** 2.).sum(dim=['x', 'y', 'z'])
            Evt = 0.5 * rho * dx * dy * dz * (ds.V ** 2.).sum(dim=['x', 'y', 'z'])
            Ewt = 0.5 * rho * dx * dy * dz * (ds.W ** 2.).sum(dim=['x', 'y', 'z'])
            ds_new['Eht'] = Eht
            ds_new['Eut'] = Eut
            ds_new['Evt'] = Evt
            ds_new['Ewt'] = Ewt

            self.cache = ds_new
            self.to_file()

        return self.cache


@xr.register_dataset_accessor('tensor')
class Tensor(DerivedDataset):

    # def __init__(self, xr_obj):
    #     super(Stats, self).__init__(xr_obj)

    def __call__(self, tag='', reprocess=False):
        super().__call__(tag='tensor', reprocess=reprocess)

        if self.cache is None:
            ds = self.parent
            ds_new = xr.Dataset(coords=ds.coords,attrs=ds.attrs)

            print('Calculating tensor')
            # j = jacobianConv(ds.U, ds.V, ds.W, dx, dy, dz, sigma=1.5)
            #print(ds.U)
            j = jacobian(ds.U.values, ds.V.values, ds.W.values, ds.attrs['piv_step_frame'], ds.attrs['dx'], ds.attrs['dy'], ds.attrs['dz'])
            #j = j.compute()
            ds_new['jacobian'] = xr.DataArray(j,dims=['time', 'x', 'y', 'z', 'comp', 'dims'])
            ds_new['jacobianNorm'] = np.sqrt((ds_new['jacobian'] ** 2.).sum(dim=['comp', 'dims']))
            jT = ds_new.jacobian.transpose('time', 'x', 'y', 'z', 'dims', 'comp').values
            ds_new['strainTensor'] = (ds_new.jacobian + jT) / 2.
            ds_new['vorticityTensor'] = (ds_new.jacobian - jT) / 2.
            ds_new['strainTensorNorm'] = np.sqrt((ds_new.strainTensor ** 2.).sum(dim=['comp', 'dims']))
            ds_new['vorticityTensorNorm'] = np.sqrt((ds_new.vorticityTensor ** 2.).sum(dim=['comp', 'dims']))

            ds_new['dudx'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 0, 0])
            ds_new['dvdy'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 1, 1])
            ds_new['dwdz'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 2, 2])
            ds_new['divergence'] = ds_new['dudx'] + ds_new['dvdy'] + ds_new['dwdz']

            #print(ds_new['divergence'])
            #
            ds_new['omx'] = ds_new['vorticityTensor'][:, :, :, :, 2, 1] * 2.
            ds_new['omy'] = ds_new['vorticityTensor'][:, :, :, :, 0, 2] * 2.
            ds_new['omz'] = ds_new['vorticityTensor'][:, :, :, :, 1, 0] * 2.

            ds_new['divNorm'] = ds_new['divergence'] / ds_new['jacobianNorm']
            ds_new['divNorm_mean'] = np.mean(ds_new['divNorm'])
            ds_new['divNorm_std'] = np.std(ds_new['divNorm'])
            delta = (ds.attrs['dx'] * ds.attrs['dy'] * ds.attrs['dz']) ** (1./3.)
            dpx = (ds.attrs['pdx'] * ds.attrs['pdy'] * ds.attrs['pdz']) ** (1./3.)
            delta_px = delta / dpx            
            dt = ds.attrs['piv_step_ensemble']
            print(dt)
            print(delta_px)
            
            ds_new['divRMS'] = np.mean(ds_new['divergence'] ** 2.) ** 0.5 * dt
            ds_new['velocityError'] = ds_new['divRMS']/((3./(2.*delta_px**2.))**0.5)
#            print(np.percentile(ds_new['vorticityTensorNorm'].data.ravel(),99.))
#           print(ds_new['divRMS'])
#           print(ds_new['divNorm_mean'])
            pct = np.percentile(np.ravel(ds_new['vorticityTensorNorm'].data),99.)
            bob = float(ds_new['divRMS']/dt/pct)
            ds_new['vorticityError'] = bob
            print(bob)
            
            print('saving')
            self.cache = ds_new
            self.to_file()

        return self.cache

class TensorDask(DerivedDataset):

    # def __init__(self, xr_obj):
    #     super(Stats, self).__init__(xr_obj)

    def __call__(self, tag='', reprocess=False):
        super().__call__(tag='tensor', reprocess=reprocess)

        if self.cache is None:
            ds = self.parent
            ds_new = xr.Dataset(coords=ds.coords,attrs=ds.attrs)

            print('Calculating tensor')
            # j = jacobianConv(ds.U, ds.V, ds.W, dx, dy, dz, sigma=1.5)
            j = jacobian(ds.U.data, ds.V.data, ds.W.data, ds.attrs['piv_step_frame'], ds.attrs['dx'], ds.attrs['dy'], ds.attrs['dz'])
            j = j.compute()
            ds_new['jacobian'] = (['time', 'x', 'y', 'z', 'comp', 'dims'], j)
            ds_new['jacobianNorm'] = da.sqrt((ds_new['jacobian'] ** 2.).sum(dim=['comp', 'dims']))
            jT = ds_new.jacobian.transpose('time', 'x', 'y', 'z', 'dims', 'comp')#.values
            ds_new['strainTensor'] = (ds_new.jacobian + jT) / 2.
            ds_new['vorticityTensor'] = (ds_new.jacobian - jT) / 2.
            ds_new['strainTensorNorm'] = da.sqrt((ds_new.strainTensor ** 2.).sum(dim=['comp', 'dims']))
            ds_new['vorticityTensorNorm'] = da.sqrt((ds_new.vorticityTensor ** 2.).sum(dim=['comp', 'dims']))

            ds_new['dudx'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 0, 0])
            ds_new['dvdy'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 1, 1])
            ds_new['dwdz'] = (['time', 'x', 'y', 'z'], j[:, :, :, :, 2, 2])
            ds_new['divergence'] = ds_new['dudx'] + ds_new['dvdy'] + ds_new['dwdz']

            print(ds_new['divergence'])
            #
            ds_new['vorticity'] = (['time','x', 'y', 'z', 'comp'], da.stack((ds_new['vorticityTensor'][:, :, :, :, 2, 1],
                                                                       ds_new['vorticityTensor'][:, :, :, :, 0, 2],
                                                                       ds_new['vorticityTensor'][:, :, :, :, 1, 0]), axis=-1))

            ds_new['divNorm'] = ds_new['divergence'] / ds_new['jacobianNorm']
            ds_new['divNorm_mean'] = da.mean(ds_new['divNorm'])
            ds_new['divNorm_std'] = da.std(ds_new['divNorm'])
            delta = (ds.attrs['dx'] * ds.attrs['dy'] * ds.attrs['dz']) ** (1./3.)
            dpx = (ds.attrs['pdx'] * ds.attrs['pdy'] * ds.attrs['pdz']) ** (1./3.)
            delta_px = delta / dpx
            dt = ds.attrs['piv_step_ensemble']

            ds_new['divRMS'] = da.mean((ds_new['divergence'] * dt) ** 2.) ** 0.5
            ds_new['velocityError'] = ds_new['divRMS']/((3./(2.*delta_px**2.))**0.5)
            print(da.percentile(ds_new['vorticityTensorNorm'].data.ravel(),99.))
            print(ds_new['divRMS'])
            print(ds_new['divNorm_mean'])
            ds_new['vorticityError'] = ds_new['divRMS']/dt/da.percentile(ds_new['vorticityTensorNorm'].data.ravel(),99.)

            print('saving')
            self.cache = ds_new
            self.to_file()

        return self.cache



# @xr.register_dataset_accessor('part')
# class PartitionedDataset(DerivedDataset):
#
#     def __init__(self, xr_obj):
#         super(PartitionedDataset, self).__init__(xr_obj, tag='part')
#         self._conditions = None
#
#     def __call__(self,tag='TAGME',conditions=None):
#         self.tag = 'part-' + tag
#         self._conditions = conditions
#         super().__call__()
#
#         if self.cache is None:
#             self.cache = self.parent.isel(indexers=self._conditions,drop=True)
#             self.to_netcdf()
#
#         return self.cache

    # def compute(self):
    #     return self.result.isel(indexers=self._conditions,drop=True)

    # @property
    # def result_file(self):
    #     fn = self._result_file
    #     # print(fn)
    #     return fn[:fn.rfind('.')] + '.' + self._tag + '.nc'

    # def to_netcdf(self,fn):
    #     fn = fn[:fn.rfind('.')] + '.' + self._tag + '.nc'
    #     ds_new.to_netcdf(fn)
    #     return fn





        # self.history = super.

    # @property
    # def ds(self):
    #     print(self.__class__)
    #     return self.ds



# class CachedDataset(object):
#
#     def __init__(self):
#         self._ds = None
#         self.tag = ''
#         self.mask =
#
#     def to_netcdf(self):
#         ds.attrs['history']
#         ds.attrs['callstack']
#
