# encoding: utf-8
"""
Module to set up run time parameters for Clawpack.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import shutil
import gzip

import numpy as np

from clawpack.geoclaw.surge.storm import Storm
import clawpack.clawutil as clawutil

# Shared code from parent directory
import sys
sys.path.append('..')
import shared_setrun

# Time Conversions
def days2seconds(days):
    return days * 60.0**2 * 24.0


# Scratch directory for storing topo and dtopo files:
scratch_dir = os.path.join(os.environ["CLAW"], 'geoclaw', 'scratch')


# ------------------------------
def setrun(claw_pkg='geoclaw'):

    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    # =======
    # Specify all storm parameters here
    # =======

    # Storm name
    storm_name = 'matthew'

    # Remote storm file location
    remote_storm_file = "http://ftp.nhc.noaa.gov/atcf/archive/2016/bal142016.dat.gz"

    # -----------
    # Landfall time
    # (somewhat arbitrary, defines relative time "0")
    # "Matthew caused its most destructive entry as it made landfall in Haiti on October 4"
    # datetime.datetime(YYYY, MM, DD, HH in UTC)
    landfall_time_offset = datetime.datetime(2016, 10, 4, 11)
    # -----------

    # -------------
    # Start and end time,
    # in days relative to landfall:
    # -------------
    t0 = -days2seconds(5) # These need adjusting
    tfinal = days2seconds(6)
    
    # ========
    
    # Remaining setup from shared code
    rundata = shared_setrun.generate_rundata(t0, tfinal, claw_pkg=claw_pkg)

    # ------------------------------------------------------------------
    # GeoClaw specific parameters:
    # ------------------------------------------------------------------

    # Run shared Caribbean code
    rundata = shared_setrun.generate_geo_data(rundata)

    # Run specific storm code
    data = rundata.surge_data
    data.storm_file = os.path.expandvars(os.path.join(os.getcwd(),
                                         storm_name.lower() + '.storm'))

    # Convert ATCF data to GeoClaw format
    clawutil.data.get_remote_file(remote_storm_file) 
    # Get file name without .gz suffix
    local_storm_dat = os.path.basename(remote_storm_file)[:-3]
    atcf_path = os.path.join(scratch_dir, local_storm_dat)
    # Note that the get_remote_file function does not support gzip files which
    # are not also tar files.  The following code handles this
    with gzip.open(".".join((atcf_path, 'gz')), 'rb') as atcf_file,    \
            open(atcf_path, 'w') as atcf_unzipped_file:
        atcf_unzipped_file.write(atcf_file.read().decode('ascii'))

    storm = Storm(path=atcf_path, file_format="ATCF")

    storm.time_offset = landfall_time_offset

    storm.write(data.storm_file, file_format='geoclaw')

    return rundata
    # end of function setgeo
    # ----------------------


if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    if len(sys.argv) == 2:
        rundata = setrun(sys.argv[1])
    else:
        rundata = setrun()

    rundata.write()
