# 
#  Copyright (C) 2009  Smithsonian Astrophysical Observatory
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
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


#!/usr/bin/env python
import numpy
import numpy.random
from sherpa.astro.utils import calc_energy_flux
from itertools import izip
from sherpa.utils import parallel_map
from sherpa.sim import NormalParameterSampleFromScaleMatrix, \
    NormalParameterSampleFromScaleVector

__all__ = ['calc_flux', 'sample_flux']

def calc_flux(fit, data, src, samples, method=calc_energy_flux,
              lo=None, hi=None, numcores=None):

    def evaluate(sample):
        fit.model.thawedpars = sample
        flux = method(data, src, lo, hi)
        return [flux] + list(sample)

    old_model_vals  = fit.model.thawedpars
    try:
        fluxes = parallel_map(evaluate, samples, numcores)
    finally:
        fit.model.thawedpars = old_model_vals

    return numpy.asarray(fluxes)


def sample_flux(fit, data, src, method=calc_energy_flux, correlated=False,
                num=1, lo=None, hi=None, numcores=None):
    sampler = NormalParameterSampleFromScaleVector()
    if correlated:
        sampler = NormalParameterSampleFromScaleMatrix()        
    samples = sampler.get_sample(fit, num=num)
    return calc_flux(fit, data, src, samples, method, lo, hi, numcores)
