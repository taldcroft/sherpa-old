# 
#  Copyright (C) 2011  Smithsonian Astrophysical Observatory
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


from parameter import Parameter, tinyval
from model import ArithmeticModel, modelCacher1d, CompositeModel, \
    ArithmeticFunctionModel
import numpy
import sherpa.utils.kdtree


__all__ = ('create_template_model', 'TemplateModel')



def create_template_model(modelname, names, parvals, templates):
    """
    Create a TemplateModel model class from template input


    `modelname`  - name of the template model.

    `names`      - list of strings that define the order of the 
                   named parameters.

    `parvals`    - 2-D ndarray of parameter vectors, index corresponds
                   to the spectrum in `templates`. The parameter grid.

    `templates`  - list of TableModel objects that contain a spectrum
                   at a specific parameter vector (corresponds to a row
                   in `parvals`).

    """
    # Create a list of parameters from input
    pars = []   
    for ii, name in enumerate(names):
        minimum = min(parvals[:,ii])
        maximum = max(parvals[:,ii])
        initial = parvals[:,ii][0]
        # Initial parameter value is always first parameter value listed
        par = Parameter(modelname, name, initial,
                        minimum, maximum,
                        minimum, maximum)
        pars.append(par)

    # Create the templates table from input
    return TemplateModel(modelname, pars, parvals, templates)


class TemplateModel(ArithmeticModel):

    def __init__(self, name='templatemodel', pars=(), parvals=[], templates=[]):
        self.parvals = parvals
        self.templates = templates
        for par in pars:
            self.__dict__[par.name] = par

        # Construct kdtree from parameter space
        self.tree = None

        lpars = len(pars)

        if lpars < 2:
            raise TypeError("Use tablemodel for a single template parameter")
        if lpars > 10:
            raise TypeError("Greater than 10 template parameters is unsupported")

        klass = getattr(sherpa.utils.kdtree, 'KDTree_%iDouble' % lpars)
        tree = klass()

        for ii, parval in enumerate(parvals):
            tree.add((tuple(parval), ii))        

        tree.optimize()
        self.tree = tree
        ArithmeticModel.__init__(self, name, pars)


    def fold(self, data):
        for template in self.templates:
            template.fold(data)


    def get_x(self):
        # query tree using p vector, obtain idex
        pvals, idx = self.tree.find_nearest(tuple(par.val for par in self.pars))
        return self.templates[idx].get_x()

    def get_y(self):
        # query tree using p vector, obtain idex
        pvals, idx = self.tree.find_nearest(tuple(par.val for par in self.pars))
        return self.templates[idx].get_y()


    def query(self, p):
        pvals, idx = self.tree.find_nearest(tuple(p))
        return self.templates[idx].get_y()


    def query_index(self, p):
        pvals, idx = self.tree.find_nearest(tuple(p))
        return idx


    @modelCacher1d
    def calc(self, p, x0, x1=None, *args, **kwargs):

        # query tree using p vector, obtain idex
        pvals, idx = self.tree.find_nearest(tuple(p))

        # lookup the spectrum according to index
        table_model = self.templates[idx]

        # return interpolated the spectrum according to the input grid (x0, [x1])
        return table_model(x0, x1, *args, **kwargs)
