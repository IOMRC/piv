import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage as nd

import os
import struct
import copy

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

    def __init__(self, xml_file='', reprocess=False, result_root=None, data_root=None, piv_pass=-1):
        self._xml_file = xml_file
        self._ds = None
        self._reprocess = reprocess

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
        self._result_file = self._data_root + '/' + self.run_id + '_' + self.piv_tag + '_velocity.nc'

    @property
    def result_file(self):
        return self._result_file

    def _loadgrid(self):

        with open(self._result_root + "xyz_grid_" + self.piv_tag + "_pass%i.bin" % self.grid, "rb") as f:
            x, y, z, xx, yy, zz = _read_binary_grid(f)

        # Unit conversions here should probably have been done in step3_piv.cpp
        # The spatial grid is output in pixels - convert to meters
        px = float(self._xml_root.findall('capture/dx')[0].text)  # mm/px
        py = float(self._xml_root.findall('capture/dy')[0].text)  # mm/px
        pz = float(self._xml_root.findall('reprojection/dz')[0].text)  # mm/px

        xx = xx[:, 0, 0] * px / 1000.
        yy = yy[0, :, 0] * py / 1000.
        zz = zz[0, 0, :] * pz / 1000.

        return xx, yy, zz

    def _loaddata(self, xx, yy, zz):

        frames = list()
        all_files = os.listdir(self._result_root + 'velocity/')
        for fn in sorted(all_files):
            if fn.startswith(self.piv_tag):
                frames.append(fn)

        tt = np.zeros((len(frames),)).astype(pd.tslib.Timestamp)
        ut = np.zeros((len(xx), len(yy), len(zz), len(frames)))
        vt = np.zeros((len(xx), len(yy), len(zz), len(frames)))
        wt = np.zeros((len(xx), len(yy), len(zz), len(frames)))

        for it, fn in enumerate(frames):
            try:
                print('%.0f - %s' % (it, fn))
                with open(self._result_root + "velocity/" + fn, "rb") as f:
                    x, y, z, u, v, w = _read_binary_grid(f)

                    # this should be subsequent filter rather than written to 'raw' velocity file
                    sigma = 0.5
                    u = nd.filters.gaussian_filter(u, sigma)
                    v = nd.filters.gaussian_filter(v, sigma)
                    w = nd.filters.gaussian_filter(w, sigma)

                    # convert to meters/s
                    ut[:, :, :, it] = u / 1000.
                    vt[:, :, :, it] = v / 1000.
                    wt[:, :, :, it] = w / 1000.

                tt[it] = pd.to_datetime(fn.split('_')[-1][:-4], format='%Y-%m-%d-%H-%M-%S.%f')

            except IOError:
                print('Cannot open ' + fn)
                shp = (len(xx), len(yy), len(zz))
                ut[:, :, :, it] = np.ones(shp) * np.nan
                vt[:, :, :, it] = np.ones(shp) * np.nan
                wt[:, :, :, it] = np.ones(shp) * np.nan

        return tt, ut, vt, wt

    def _append_CF_attrs(self, ds):
        ds.attrs['title'] = 'Three dimensional particle imaging velocimetry data'
        ds.attrs['institution'] = 'The University of Western Australia'
        ds.attrs['source'] = '3DPIV applied to Synthetic Aperture Imagery'
        ds.attrs['references'] = 'TBD'

    def _append_attrs(self, ds):
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
            file_exists = os.path.isfile(self._result_file)

            reprocess = not file_exists or self._reprocess

            if reprocess:
                print('Reprocessing binary data...')
                xx, yy, zz = self._loadgrid()
                tt, uu, vv, ww = self._loaddata(xx, yy, zz)

                # calculate the elapsed time from the Timestamp objects and then convert to datetime64 datatype
                to_seconds = np.vectorize(lambda x: x.seconds + x.microseconds / 1E6)
                te = to_seconds(tt - tt[0])
                tt = pd.to_datetime(tt)

                u = xr.DataArray(uu, coords=[xx, yy, zz, tt], dims=['x', 'y', 'z', 'time'],
                                 name='U', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
                v = xr.DataArray(vv, coords=[xx, yy, zz, tt], dims=['x', 'y', 'z', 'time'],
                                 name='V', attrs={'standard_name': 'sea_water_x_velocity', 'units': 'm s-1'})
                w = xr.DataArray(ww, coords=[xx, yy, zz, tt], dims=['x', 'y', 'z', 'time'],
                                 name='W', attrs={'standard_name': 'upward_sea_water_velocity', 'units': 'm s-1'})
                te = xr.DataArray(te, coords=[tt, ], dims=['time', ])

                ds_data = OrderedDict()
                ds_data['U'] = u
                ds_data['V'] = v
                ds_data['W'] = w
                ds_data['te'] = te

                ds = xr.Dataset(ds_data)
                ds.chunk({'time': 1})

                self._append_CF_attrs(ds)
                self._append_attrs(ds)
                ds.attrs['filename'] = self._result_file
                ds.to_netcdf(self._result_file)
                print('Cached ' + self._result_file)
            else:
                ds = xr.open_dataset(self._result_file,chunks={'time':1})
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
            file_exists = os.path.isfile(self.cache_file)
            reprocess = not file_exists or self._reprocess
            if not reprocess:
                _cache_ds = xr.open_dataset(self.cache_file)
                self.cache = _cache_ds

    @property
    def parent(self):
        return self._parent

    @property
    def cache_file(self):
        fn = self.parent.attrs['filename']
        fn = fn[:fn.rfind('.')] + '.' + self._tag + '.nc'
        return fn

    def to_netcdf(self):
        self.cache.attrs['filename'] = self.cache_file
        self.cache.to_netcdf(self.cache_file)


# from .accessor import register_result_accessor


def jacobian(u, v, w, dx, dy, dz):

    du = np.gradient(u, dx, dy, dz, axis=(0, 1, 2))
    dv = np.gradient(v, dx, dy, dz, axis=(0, 1, 2))
    dw = np.gradient(w, dx, dy, dz, axis=(0, 1, 2))

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
            self.to_netcdf()

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
            j = jacobian(ds.U, ds.V, ds.W, ds.dx, ds.dy, ds.dz)
            ds_new['jacobian'] = (['x', 'y', 'z', 'time', 'comp', 'dims'], j)
            ds_new['jacobianNorm'] = np.sqrt((ds_new.jacobian ** 2.).sum(dim=['comp', 'dims']))
            jT = ds_new.jacobian.transpose('x', 'y', 'z', 'time', 'dims', 'comp').values
            ds_new['strainTensor'] = (ds_new.jacobian + jT) / 2.
            ds_new['vorticityTensor'] = (ds_new.jacobian - jT) / 2.
            ds_new['strainTensorNorm'] = np.sqrt((ds_new.strainTensor ** 2.).sum(dim=['comp', 'dims']))
            ds_new['vorticityTensorNorm'] = np.sqrt((ds_new.vorticityTensor ** 2.).sum(dim=['comp', 'dims']))
            # ds_new['jacobianP'] = (['x', 'y', 'z', 'time'], -np.trace(ds_new['jacobian'], axis1=-2, axis2=-1))
            # ds_new['jacobianQ'] = 0.5 * (ds_new['vorticityTensorNorm'] ** 2. - ds_new['strainTensorNorm'] ** 2.)
            # ds_new['jacobianR'] = (['x', 'y', 'z', 'time'], -np.linalg.det(np.nan_to_num(ds_new['jacobian'].values)))

            # j = jacobian(U, V, W, dx, dy, dz)
            # # jDA = xr.DataArray(j,coords={'x':})
            # # ds['jacobian'] = (['x', 'y', 'z', 'time', 'comp', 'dims'], )
            # ds['jacobianNorm'] = (['x', 'y', 'z', 'time'],np.sqrt(np.sum(np.sum(j ** 2.,axis=-1),axis=-1)))
            # jT = j.transpose(0, 1, 2, 3, 5, 4)
            # strain = (j + jT)/2.
            # vort = (j - jT)/2.
            # ds['strainTensorNorm'] = (['x', 'y', 'z', 'time'],np.sqrt(np.sum(np.sum(strain ** 2.,axis=-1),axis=-1)))
            # ds['vorticityTensorNorm'] = (['x', 'y', 'z', 'time'],np.sqrt(np.sum(np.sum(vort ** 2.,axis=-1),axis=-1)))
            # ds['jacobianP'] = (['x', 'y', 'z', 'time'],-np.trace(j,axis1=-2,axis2=-1))
            # ds['jacobianQ'] = 0.5 * (ds['vorticityTensorNorm'] ** 2. - ds['strainTensorNorm'] ** 2.)
            # ds['jacobianR'] = (['x', 'y', 'z', 'time'], -np.linalg.det(np.nan_to_num(j)))
            #
            # Qcrit = copy.deepcopy(ds_new['jacobianQ'].values)
            # Qcrit[Qcrit < 0.] = 0.
            # ds_new['Qcriterion'] = (['x', 'y', 'z', 'time'], Qcrit)
            #
            # Deltacrit = copy.deepcopy(((ds_new.jacobianQ / 3.) ** 3. + (ds_new.jacobianQ / 2.) ** 2.).values)
            # Deltacrit[Qcrit < 0.] = 0.
            # ds_new['Deltacriterion'] = (['x', 'y', 'z', 'time'], Qcrit)
            #
            # L2calc = np.nan_to_num((ds_new['strainTensor'] ** 2. + ds_new['vorticityTensor'] ** 2.))
            # L2 = np.linalg.eigvals(L2calc)
            # L2 = np.sort(L2, axis=-1)
            # ds_new['L2criterion'] = (['x', 'y', 'z', 'time'], np.squeeze(L2[:, :, :, :, 1]))
            #
            # eigVals = np.linalg.eigvals(np.nan_to_num(j))
            # swirl = np.max(np.imag(eigVals), axis=-1)
            # ds_new['Swirlcriterion'] = (['x', 'y', 'z', 'time'], swirl)

            # ds['dudx'] = ds.jacobian.sel(comp=0, dims=0)
            # ds['dvdy'] = ds.jacobian.sel(comp=1, dims=1)
            # ds['dwdz'] = ds.jacobian.sel(comp=2, dims=2)

            # ds_new['dudx'] = (['x', 'y', 'z', 'time'], j[:, :, :, :, 0, 0])
            # ds_new['dvdy'] = (['x', 'y', 'z', 'time'], j[:, :, :, :, 1, 1])
            # ds_new['dwdz'] = (['x', 'y', 'z', 'time'], j[:, :, :, :, 2, 2])
            #
            # # ds['omx'] = ds['vorticityTensor'][:, :, :, :, 1, 2] * 2.
            # # ds['omy'] = ds['vorticityTensor'][:, :, :, :, 0, 2] * 2.
            # # ds['omz'] = ds['vorticityTensor'][:, :, :, :, 0, 1] * 2.
            #
            # ds_new['om'] = (['x', 'y', 'z', 'time', 'comp'], np.stack((ds_new['vorticityTensor'][:, :, :, :, 2, 1],
            #                                                            ds_new['vorticityTensor'][:, :, :, :, 0, 2],
            #                                                            ds_new['vorticityTensor'][:, :, :, :, 1, 0]), axis=-1))
            #
            # for ii, c in enumerate(['u', 'v', 'w']):
            #     for jj, d in enumerate(['x', 'y', 'z']):
            #         ds_new['om' + d + 'd' + c + 'd' + d] = ds_new['om'][:, :, :, :, jj] * ds_new.jacobian[:, :, :, :, ii, jj]
            #
            #         # ds['d2' + c + 'd' + d + '2']=(['x', 'y', 'z', 'time'],np.gradient(np.gradient(ds[c],dd,axis=jj),dd,axis=jj))#Not a good wayt to do laplacian
            #         # Calculate laplacian by filtering with a 2nd order gaussian kernel
            #
            # for ii, c in enumerate(['x', 'y', 'z']):
            #     for jj, d in enumerate(['x', 'y', 'z']):
            #         if d == 'x':
            #             dd = ds.dx
            #         elif d == 'y':
            #             dd = ds.dy
            #         elif d == 'z':
            #             dd = ds.dz
            #         else:
            #             dd == 0
            #         ds_new['d2om' + c + 'd' + d + '2'] = (['x', 'y', 'z', 'time'],
            #                                           ndimage.gaussian_filter1d(ds_new['om'][:, :, :, :, ii], axis=jj,
            #                                                                     sigma=1.5, order=2, mode='reflect') / (
            #                                                       dd ** 2.))
            print('saving')
            self.cache = ds_new
            self.to_netcdf()

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
