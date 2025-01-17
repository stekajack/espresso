# Copyright (C) 2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest as ut
import importlib_wrapper
import numpy as np

try:
    import pint  # pylint: disable=unused-import
except ImportError:
    tutorial = importlib_wrapper.MagicMock()
    skipIfMissingFeatures = ut.skip(
        "Python module pint not available, skipping test!")
else:
    tutorial, skipIfMissingFeatures = importlib_wrapper.configure_and_import(
        "@TUTORIALS_DIR@/constant_pH/constant_pH.py", script_suffix="ideal",
        USE_WCA=False, USE_ELECTROSTATICS=False, NUM_PHS=10)


@skipIfMissingFeatures
class Tutorial(ut.TestCase):
    system = tutorial.system

    def test(self):
        ref_values = 1. / (1 + 10**(tutorial.pKa - tutorial.pHs))
        sim_values = tutorial.av_alpha
        sim_values_error = tutorial.err_alpha
        # test alpha +/- 0.02 and standard error of alpha less than 0.02
        np.testing.assert_allclose(sim_values, ref_values, rtol=0., atol=0.02)
        np.testing.assert_array_less(sim_values_error, 0.02)


if __name__ == "__main__":
    ut.main()
