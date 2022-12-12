#!/usr/bin/env bash
#
# Copyright (C) 2018-2022 The ESPResSo project
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
#

abort() {
  echo "An error occurred in runner.sh, exiting now" >&2
  echo "Command that failed: ${BASH_COMMAND}" >&2
  exit 1
}

trap abort EXIT
set -e

# manage configuration files with different features
configs="maxset.hpp default.hpp empty.hpp"
cp ../maintainer/configs/empty.hpp .
cp ../src/config/myconfig-default.hpp default.hpp
cp ../maintainer/configs/maxset.hpp .

# process configuration files
for config in ${configs}; do
  # add minimal features for the benchmarks to run
  sed -i '1 i\#define ELECTROSTATICS\n#define LENNARD_JONES\n#define MASS\n' "${config}"
  # remove checks
  sed -ri "/#define\s+ADDITIONAL_CHECKS/d" "${config}"
done

cat > benchmarks.csv << EOF
"config","script","arguments","cores","mean","ci","nsteps","duration","label"
EOF

# run benchmarks
for config in ${configs}; do
  echo "### ${config}" | tee -a benchmarks.log
  cp ${config} myconfig.hpp
  rm -rf src/ maintainer/
  cmake -D ESPRESSO_BUILD_BENCHMARKS=ON -D ESPRESSO_TEST_TIMEOUT=1200 -D ESPRESSO_BUILD_WITH_CUDA=OFF -D ESPRESSO_BUILD_WITH_CCACHE=OFF ..
  make -j$(nproc)
  rm -f benchmarks.csv.part
  touch benchmarks.csv.part
  make benchmark 2>&1 | tee -a benchmarks.log
  sed -ri "s/^/\"$(basename ${config})\",/" benchmarks.csv.part
  cat benchmarks.csv.part >> benchmarks.csv
done

rm benchmarks.csv.part

trap : EXIT
