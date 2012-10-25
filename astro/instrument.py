# 
#  Copyright (C) 2010  Smithsonian Astrophysical Observatory
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import numpy
from sherpa.utils.err import InstrumentErr, DataErr
from sherpa.models.model import ArithmeticFunctionModel, NestedModel, \
    ArithmeticModel, CompositeModel, Model

from sherpa.utils import NoNewAttributesAfterInit
from sherpa.data import BaseData
from sherpa.astro.data import DataARF, DataRMF, DataPHA, _notice_resp
from sherpa.utils import sao_fcmp, sum_intervals, sao_arange
from sherpa.astro.utils import compile_energy_grid
from itertools import izip

_tol = numpy.finfo(numpy.float32).eps

__all__ = ('RMFModel', 'ARFModel', 'MultiResponseSumModel', 'PileupRMFModel',
           'RMF1D', 'ARF1D',
           'Response1D', 'MultipleResponse1D','PileupResponse1D')


class RMFModel(CompositeModel, ArithmeticModel):

    def __init__(self, rmf, model, pha=None, arf=None):
        _notice_resp(None, arf, rmf)
        self.channel = sao_arange(1, rmf.detchans)  # sao_arange is inclusive
        self.mask = numpy.ones(rmf.detchans, dtype=bool)
        self.rmf = rmf
        self.arf = arf
        self.pha = pha
        self.elo = None; self.ehi = None  # Energy space
        self.lo = None; self.hi = None    # Wavelength space

        # EXPERIMENTAL voodoo
        if arf is None and isinstance(model, ARFModel):
            self.arf = model.arf
            if model.rmf is None:
                model.rmf = self.rmf

        #self.model = NestedModel.wrapobj(model)
        self.model = model
        self.otherargs = None
        self.otherkwargs = None
        self.pars = ()
        CompositeModel.__init__(self,
                                ('%s(%s)' % ('apply_rmf', self.model.name)),
                                (self.model,))
        self._get_otherargs()


    def _get_otherargs(self):
        elo, ehi = self.rmf.get_indep()
        args = ()

        # PHA <=> RMF case
        pha = self.pha
        if (self.arf is None and pha is not None and 
            pha.bin_lo is not None and pha.bin_hi is not None):
            if len(pha.bin_lo) != len(self.rmf.energ_lo):
                args = pha._get_ebins(group=False)

        self.elo, self.ehi = elo, ehi
        self.lo, self.hi = DataPHA._hc/self.ehi, DataPHA._hc/self.elo

        # Compare malformed grids in energy space
        self.otherargs = ((elo,ehi), args)


    def startup(self):
        pha = self.pha
        if pha is not None:
            self.channel = pha.get_noticed_channels()
            self.mask = pha.get_mask()
            if numpy.iterable(pha.mask):
                _notice_resp(self.channel, self.arf, self.rmf)

        self._get_otherargs()
        self.model.startup()
        CompositeModel.startup(self)


    def teardown(self):
        pha = self.pha
        rmf = self.rmf
        self.channel = sao_arange(1, rmf.detchans)
        self.mask = numpy.ones(rmf.detchans, dtype=bool)

        if pha is not None:
            if numpy.iterable(pha.mask):
                _notice_resp(None, self.arf, rmf)

        self._get_otherargs()
        self.model.teardown()
        CompositeModel.teardown(self)


    def _check_for_user_grid(self, x):
        return (len(self.channel) != len(x) or
                not (sao_fcmp(self.channel, x, _tol)==0).all())


    def _startup_user_grid(self, x):
        # fit() never comes in here b/c it calls startup()

        self.mask = numpy.zeros(self.rmf.detchans, dtype=bool)
        self.mask[numpy.searchsorted(self.channel, x)]=True

        _notice_resp(x, self.arf, self.rmf)

        self._get_otherargs()
        if hasattr(self.model, '_get_otherargs'):
            self.model._get_otherargs()


    def _teardown_user_grid(self):
        # fit() never comes in here b/c it calls startup()

        self.mask = numpy.ones(self.rmf.detchans, dtype=bool)

        _notice_resp(None, self.arf, self.rmf)

        self._get_otherargs()
        if hasattr(self.model, '_get_otherargs'):
            self.model._get_otherargs()


    def _calc(self, p, xlo, xhi):
        # Evaluate source model on RMF energy/wave grid OR
        # model.calc --> arf fold / source model
        src = self.model.calc(p, xlo, xhi)

        # rmf_fold
        return self.rmf.apply_rmf(src, *self.otherargs)


    def calc(self, p, x, xhi=None, *args, **kwargs):
        # x is noticed/full channels here

        pha = self.pha
        user_grid = False
        try:

            if self._check_for_user_grid(x):
                user_grid = True
                self._startup_user_grid(x)

            xlo, xhi = self.elo, self.ehi
            if pha is not None and pha.units == 'wavelength':
                xlo, xhi = self.lo, self.hi

            vals = self._calc(p, xlo, xhi)
            if self.mask is not None:
                vals = vals[self.mask]

        finally:
            if user_grid:
                self._teardown_user_grid()

        return vals


class ARFModel(CompositeModel, ArithmeticModel):

    def __init__(self, arf, model, pha=None, rmf=None):
        _notice_resp(None, arf, rmf)
        self.size = len(arf.specresp)

        # Channel is needed for ARF only analysis,
        # FIXME should grating PHA pass bin_lo, bin_hi instead
        #       of channels?
        self.channel = sao_arange(1, self.size)

        self.rmf = rmf
        self.arf = arf
        self.pha = pha
        self.elo = None; self.ehi = None  # Energy space
        self.lo = None; self.hi = None    # Wavelength space

        self.model = model
        self.otherargs = None
        self.otherkwargs = None
        self.pars = ()
        CompositeModel.__init__(self,  
                                ('%s(%s)' % ('apply_arf', self.model.name)),
                                (self.model,))
        self._get_otherargs()


    def _get_otherargs(self):
        elo, ehi = self.arf.get_indep()
        args = ()

        # PHA <=> ARF case
        pha = self.pha
        if (pha is not None and
            pha.bin_lo is not None and pha.bin_hi is not None):
            if len(pha.bin_lo) != len(self.arf.energ_lo):
                args = pha._get_ebins(group=False)

        # RMF <=> ARF case
        rmf = self.rmf
        if rmf is not None:
            if len(self.arf.energ_lo) != len(rmf.energ_lo):
                args = rmf.get_indep()

        self.elo, self.ehi = elo, ehi
        self.lo, self.hi = DataPHA._hc/self.ehi, DataPHA._hc/self.elo

        # Compare malformed grids in energy space
        self.otherargs = ((elo,ehi), args)


    def startup(self):
        pha = self.pha
        if self.rmf is None and pha is not None:
            self.channel = pha.get_noticed_channels()
            self.size = len(self.channel)
            if numpy.iterable(pha.mask):
                self.arf.notice(pha.get_mask())

        self._get_otherargs()
        self.model.startup()
        CompositeModel.startup(self)


    def teardown(self):
        pha = self.pha
        self.size = len(self.arf.specresp)
        self.channel = sao_arange(1, self.size)
        if self.rmf is None and pha is not None:
            if numpy.iterable(pha.mask):
                self.arf.notice(None)

        self._get_otherargs()
        self.model.teardown()
        CompositeModel.teardown(self)


    def _check_for_user_grid(self, xlo, lo, hi=None):

        # energy/wave
        x = xlo
        if hi is None:
            # channel
            x = self.channel

        return (self.rmf is None and
                (self.size != len(lo) or
                 not (sao_fcmp(x, lo, _tol)==0).all()))            


    def _startup_user_grid(self, elo, lo, hi=None):
        # fit() never comes in here b/c it calls startup()

        if self.rmf is None:
            # energy!
            x = elo
            if hi is None:
                # channel
                x = self.channel

            if hi is not None and lo[0] > lo[-1] and hi[0] > hi[-1]:
                lo = DataPHA._hc/lo

            mask = numpy.zeros(self.size, dtype=bool)
            mask[numpy.searchsorted(x, lo)]=True
            self.arf.notice(mask)
            self._get_otherargs()
            #print "startup_user_grid..."


    def _teardown_user_grid(self):
        # fit() never comes in here b/c it calls startup()

        if self.rmf is None:
            self.arf.notice(None)
            self._get_otherargs()
            #print "teardown_user_grid..."


    def _calc(self, p, xlo, xhi):
        # Evaluate source model on ARF energy/wave grid
        # model.calc --> source model
        src = self.model.calc(p, xlo, xhi)

        # arf_fold
        return self.arf.apply_arf(src, *self.otherargs)


    def calc(self, p, lo, hi=None, *args, **kwargs):
        user_grid = False
        try:

            if self._check_for_user_grid(self.elo, lo, hi):
                user_grid = True
                self._startup_user_grid(self.elo, lo, hi)

            # determine the working quantity from inputs
            xlo, xhi = self.elo, self.ehi             # energy
            if (self.pha is not None and self.pha.units == 'wavelength'):
                xlo, xhi = self.lo, self.hi           # wave
            elif (hi is not None and
                  lo[0] > lo[-1] and hi[0] > hi[-1]):
                xlo, xhi = self.lo, self.hi           # wave

            vals = self._calc(p, xlo, xhi)

        finally:
            if user_grid:
                self._teardown_user_grid()

        return vals


class ARF1D(NoNewAttributesAfterInit):

    def __init__(self, arf, pha=None, rmf=None):
        self._arf = arf
        self._rmf = rmf
        self._pha = pha
        NoNewAttributesAfterInit.__init__(self)

    def __getattr__(self, name):
        arf = None
        try:
            arf = ARF1D.__getattribute__(self, '_arf')
        except:
            pass

        if name in ('_arf', '_arf', '_pha'):
            return self.__dict__[name]

        if arf is not None:
            return DataARF.__getattribute__(arf, name)

        return ARF1D.__getattribute__(self, name)

    def __setattr__(self, name, val):
        arf = None
        try:
            arf = ARF1D.__getattribute__(self, '_arf')
        except:
            pass

        if arf is not None and hasattr(arf, name):
            DataARF.__setattr__(arf, name, val)
        else:
            NoNewAttributesAfterInit.__setattr__(self, name, val)

    def __dir__(self):
        return dir(self._arf)

    def __str__(self):
        return str(self._arf)

    def __repr__(self):
        return repr(self._arf)

    def __call__(self, model):
        arf = self._arf
        pha = self._pha

        # Automatically add exposure time to source model
        if pha is not None and pha.exposure is not None:
            model = pha.exposure * model
        elif arf.exposure is not None:
            model = arf.exposure * model

        return ARFModel(arf, model, pha, self._rmf)


class RMF1D(NoNewAttributesAfterInit):

    def __init__(self, rmf, pha=None, arf=None):
        self._rmf = rmf
        self._arf = arf
        self._pha = pha
        NoNewAttributesAfterInit.__init__(self)

    def __getattr__(self, name):
        rmf = None
        try:
            rmf = RMF1D.__getattribute__(self, '_rmf')
        except:
            pass

        if name in ('_arf', '_rmf', '_pha'):
            return self.__dict__[name]

        if rmf is not None:
            return DataRMF.__getattribute__(rmf, name)

        return RMF1D.__getattribute__(self, name)


    def __setattr__(self, name, val):
        rmf = None
        try:
            rmf = RMF1D.__getattribute__(self, '_rmf')
        except:
            pass

        if rmf is not None and hasattr(rmf, name):
            DataRMF.__setattr__(rmf, name, val)
        else:
            NoNewAttributesAfterInit.__setattr__(self, name, val)

    def __dir__(self):
        return dir(self._rmf)

    def __str__(self):
        return str(self._rmf)

    def __repr__(self):
        return repr(self._rmf)

    def __call__(self, model):
        arf = self._arf
        pha = self._pha

        # Automatically add exposure time to source model for RMF-only analysis
        if type(model) not in (ARFModel,):

            if pha is not None and pha.exposure is not None:
                model = pha.exposure * model
            elif arf is not None and arf.exposure is not None:
                model = arf.exposure * model

        return RMFModel(self._rmf, model, pha, arf)


class Response1D(NoNewAttributesAfterInit):

    def __init__(self, pha):
        self.pha = pha
        arf_data, rmf_data = self.pha.get_response()
        if arf_data is None and rmf_data is None:
            raise DataErr('norsp', pha.name)

        NoNewAttributesAfterInit.__init__(self)

    def __call__(self, model):
        pha = self.pha
        arf_data, rmf_data = self.pha.get_response()

        if arf_data is None and rmf_data is None:
            raise DataErr('norsp', pha.name)

        if arf_data is not None:
            model = ARF1D(arf_data, pha, rmf_data)(model)
        if rmf_data is not None:
            model = RMF1D(rmf_data, pha, arf_data)(model)

        return model


class ResponseNestedModel(Model):

    def __init__(self, arf=None, rmf=None):
        self.arf = arf
        self.rmf = rmf

        name = ''
        if arf is not None and rmf is not None:
            name = 'apply_rmf(apply_arf('
        elif arf is not None:
            name = 'apply_arf('
        elif rmf is not None:
            name = 'apply_rmf('
        Model.__init__(self, name)


    def calc(self, p, *args, **kwargs):
        arf = self.arf
        rmf = self.rmf

        if arf is not None and rmf is not None:
            return rmf.apply_rmf(arf.apply_arf(*args, **kwargs))
        elif self.arf is not None:
            return arf.apply_arf(*args, **kwargs)

        return rmf.apply_rmf(*args, **kwargs)


class MultiResponseSumModel(CompositeModel, ArithmeticModel):

    def __init__(self, source, pha):
        self.channel = pha.channel
        self.mask = numpy.ones(len(pha.channel), dtype=bool)
        self.pha = pha
        self.source = source
        self.elo = None; self.ehi = None
        self.lo = None; self.hi = None
        self.table = None
        self.orders = None

        models = []
        grid = []

        for id in pha.response_ids:
            arf, rmf = pha.get_response(id)

            if arf is None and rmf is None:
                raise DataErr('norsp', pha.name)

            m = ResponseNestedModel(arf, rmf)
            indep = None

            if arf is not None:
                indep = arf.get_indep()

            if rmf is not None:
                indep = rmf.get_indep()

            models.append(m)
            grid.append(indep)

        self.models = models
        self.grid = grid

        name = '%s(%s)' % (type(self).__name__,
                           ','.join(['%s(%s)' % (m.name, source.name)
                                     for m in models]))
        CompositeModel.__init__(self, name, (source,))


    def _get_noticed_energy_list(self):
        grid = []
        for id in self.pha.response_ids:
            arf, rmf = self.pha.get_response(id)
            indep = None
            if arf is not None:
                indep = arf.get_indep()
            elif rmf is not None:
                indep = rmf.get_indep()
            grid.append(indep)

        self.elo, self.ehi, self.table = compile_energy_grid(grid)
        self.lo, self.hi = DataPHA._hc/self.ehi, DataPHA._hc/self.elo


    def startup(self):
        pha = self.pha
        if numpy.iterable(pha.mask):
            pha.notice_response(True)
        self.channel = pha.get_noticed_channels()
        self.mask = pha.get_mask()
        self._get_noticed_energy_list()
        CompositeModel.startup(self)


    def teardown(self):
        pha = self.pha
        if numpy.iterable(pha.mask):
            pha.notice_response(False)
        self.channel = pha.channel
        self.mask = numpy.ones(len(pha.channel), dtype=bool)
        self.elo = None; self.ehi = None; self.table = None
        self.lo = None; self.hi = None
        CompositeModel.teardown(self)


    def _check_for_user_grid(self, x, xhi=None):
        return (len(self.channel) != len(x) or
                not (sao_fcmp(self.channel, x, _tol)==0).all())


    def _startup_user_grid(self, x, xhi=None):
        # fit() never comes in here b/c it calls startup()
        pha = self.pha
        self.mask = numpy.zeros(len(pha.channel), dtype=bool)
        self.mask[numpy.searchsorted(pha.channel, x)]=True
        pha.notice_response(True, x)
        self._get_noticed_energy_list()


    def _teardown_user_grid(self):
        # fit() never comes in here b/c it calls startup()
        pha = self.pha
        self.mask = numpy.ones(len(pha.channel), dtype=bool)
        pha.notice_response(False)
        self.elo = None; self.ehi = None; self.table = None
        self.lo = None; self.hi = None


    def calc(self, p, x, xhi=None, *args, **kwargs):
        pha = self.pha

        user_grid = False
        try:

            if self._check_for_user_grid(x, xhi):
                user_grid = True
                self._startup_user_grid(x, xhi)

            # Slow
            if self.table is None:
                # again, fit() never comes in here b/c it calls startup()
                src = self.source
                vals = []
                for model, args in izip(self.models, self.grid):
                    elo,ehi = lo,hi = args
                    if pha.units == 'wavelength':
                        lo = DataPHA._hc / ehi
                        hi = DataPHA._hc / elo
                    vals.append(model(src(lo, hi)))
                self.orders = vals
            # Fast
            else:
                xlo,xhi = self.elo, self.ehi
                if pha.units == 'wavelength':
                    xlo, xhi = self.lo, self.hi

                src = self.source(xlo, xhi)  # hi-res grid of all ARF grids

                # Fold summed intervals through the associated response.
                self.orders = \
                    [model(sum_intervals(src, interval[0], interval[1]))
                     for model, interval in izip(self.models, self.table)]

            vals = sum(self.orders)
            if self.mask is not None:
                vals = vals[self.mask]

        finally:
            if user_grid:
                self._teardown_user_grid()


        return vals


class MultipleResponse1D(Response1D):

    def __call__(self, model):
        pha = self.pha

        pha.notice_response(False)

        model = MultiResponseSumModel(model, pha)

        if pha.exposure:
            model = pha.exposure * model

        return model


class PileupRMFModel(CompositeModel, ArithmeticModel):

    def __init__(self, rmf, model, pha=None):
        self.pha = pha
        self.channel = sao_arange(1, rmf.detchans)  # sao_arange is inclusive
        self.mask = numpy.ones(rmf.detchans, dtype=bool)
        self.rmf = rmf
        self.elo, self.ehi = rmf.get_indep()
        self.lo, self.hi = DataPHA._hc/self.ehi, DataPHA._hc/self.elo
        self.model = model
        self.otherargs = None
        self.otherkwargs = None
        self.pars = ()
        CompositeModel.__init__(self,           
                                ('%s(%s)' % ('apply_rmf', self.model.name)),
                                (self.model,))

    def startup(self):
        pha = self.pha
        pha.notice_response(False)         
        self.channel = pha.get_noticed_channels()
        self.mask = pha.get_mask()
        self.model.startup()
        CompositeModel.startup(self)


    def teardown(self):
        pha = self.pha
        rmf = self.rmf
        self.channel = sao_arange(1, rmf.detchans)
        self.mask = numpy.ones(rmf.detchans, dtype=bool)
        self.model.teardown()
        CompositeModel.teardown(self)


    def _check_for_user_grid(self, x):
        return (len(self.channel) != len(x) or
                not (sao_fcmp(self.channel, x, _tol)==0).all())


    def _startup_user_grid(self, x):
        # fit() never comes in here b/c it calls startup()
        self.mask = numpy.zeros(self.rmf.detchans, dtype=bool)
        self.mask[numpy.searchsorted(self.pha.channel, x)]=True


    def _calc(self, p, xlo, xhi):
        # Evaluate source model on RMF energy/wave grid OR
        # model.calc --> pileup_model
        src = self.model.calc(p, xlo, xhi)

        # rmf_fold
        return self.rmf.apply_rmf(src)


    def calc(self, p, x, xhi=None, **kwargs):
        pha = self.pha
        # x is noticed/full channels here

        user_grid = False
        try:
            if self._check_for_user_grid(x):
                user_grid = True
                self._startup_user_grid(x)

            xlo, xhi = self.elo,self.ehi
            if pha is not None and pha.units == 'wavelength':
                xlo, xhi = self.lo, self.hi

            vals = self._calc(p, xlo, xhi)
            if self.mask is not None:
                vals = vals[self.mask]

        finally:
            if user_grid:
                self.mask = numpy.ones(self.rmf.detchans, dtype=bool)

        return vals


class PileupResponse1D(NoNewAttributesAfterInit):

    def __init__(self, pha, pileup_model):
        self.pha = pha
        self.pileup_model = pileup_model
        NoNewAttributesAfterInit.__init__(self)

    def __call__(self, model):
        pha = self.pha
        # clear out any previous response filter
        pha.notice_response(False)

        arf, rmf = pha.get_response()
        err_msg = None

        if arf is None and rmf is None:
            raise DataErr('norsp', pha.name)

        if arf is None:
            err_msg = 'does not have an associated ARF'
        elif pha.exposure is None:
            err_msg = 'does not specify an exposure time'

        if err_msg:
            raise InstrumentErr('baddata', pha.name, err_msg)

        # Currently, the response is NOT noticed using pileup

        # ARF convolution done inside ISIS pileup module
        # on finite grid scale
        model = model.apply(self.pileup_model, pha.exposure, arf.energ_lo,
                            arf.energ_hi, arf.specresp, model)

        if rmf is not None:
            model = PileupRMFModel(rmf, model, pha)
        return model
