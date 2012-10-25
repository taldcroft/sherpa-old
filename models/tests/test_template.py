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


from sherpa.models import TableModel, Gauss1D
from sherpa.models.template import TemplateModel, create_template_model
from sherpa.utils import SherpaTest, SherpaTestCase
import numpy

class test_template(SherpaTestCase):

    def setUp(self):
        self.num = 4
        self.ncoords = 100
        self.ntemplates = 2**self.num
        self.x = numpy.linspace(0.1, 5, 50)
        g1 = Gauss1D('g1')

        # create a 4-dimensional grid from 0 to 1 inclusive, shape = (16,4)
        grid = numpy.mgrid[ [slice(0,2,1) for ii in range(self.num)] ]
        grid = numpy.asarray(map(numpy.ravel, grid)).T
        coords = numpy.linspace(0.01, 6, 100)
        names = ["p%i" % i for i in range(self.num)]
        templates = []
        for ii in range(self.ntemplates):
            t = TableModel()
            g1.fwhm = numpy.random.uniform(0.5, 2.0)
            g1.pos  = numpy.random.uniform(1.0, 4.5)
            g1.ampl = numpy.random.uniform(1.0, 50.)
            t.load(coords, g1(coords))
            templates.append(t)

        self.model = create_template_model("mdl", names, grid, templates)

    def tearDown(self):
        self.model = None


    def test_template_model_evaluation(self):
        self.model.thawedpars = [0,1,0,1]
        vals = self.model(self.x)

    def test_template_query_index(self):
        expected = 5
        result = self.model.query_index([0,1,0,1])
        self.assertEqual(expected, result,
                         "Expected %s instead of %s" % (str(expected), str(result)))

    def test_template_query(self):
        result = self.model.query([0,1,0,1])



if __name__ == '__main__':

    import sherpa.models as models
    SherpaTest(models).test()
