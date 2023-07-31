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


# Scratch directory for storing topo and dtopo files:
scratch_dir = os.path.join(os.environ["CLAW"], 'geoclaw', 'scratch')


# ------------------------------
def generate_rundata(t0, tfinal, claw_pkg='geoclaw'):

    """
    Define the parameters used for running Clawpack.

    INPUT:
        t0 < tfinal 

    OUTPUT:
        rundata - object of class ClawRunData

    """

    from clawpack.clawutil import data

    assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)

    # ------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #   (or to amr2ez.data for AMR)
    # ------------------------------------------------------------------
    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.


    # -------------
    # Start and end times:
    # -------------
    clawdata.t0 = t0
    clawdata.tfinal = tfinal

    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    # Lower and upper edge of computational domain:
    clawdata.lower[0] = -88.0      # west longitude
    clawdata.upper[0] = -45.0      # east longitude

    clawdata.lower[1] = 9.0       # south latitude
    clawdata.upper[1] = 31.0      # north latitude

    # Number of grid cells:
    degree_factor = 4  # (0.25 deg,0.25 deg) ~ (25237.5 m, 27693.2 m) resolution
    clawdata.num_cells[0] = int(clawdata.upper[0] - clawdata.lower[0]) \
        * degree_factor
    clawdata.num_cells[1] = int(clawdata.upper[1] - clawdata.lower[1]) \
        * degree_factor

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    # First three are from shallow GeoClaw, fourth is friction and last 3 are
    # storm fields
    clawdata.num_aux = 3 + 1 + 3

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2

    # Restart from checkpoint file of a previous run?
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False               # True to restart from prior results
    clawdata.restart_file = 'fort.chk00006'  # File to use for restart data

    # -------------
    # Output times:
    # --------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
    # The solution at initial time t0 is always written in addition.

    clawdata.output_style = 1

    if clawdata.output_style == 1:
        # Output nout frames at equally spaced times up to tfinal:
        recurrence = 4
        clawdata.num_output_times = int((clawdata.tfinal - clawdata.t0) *
                                        recurrence / (60**2 * 24))

        clawdata.output_t0 = True  # output at initial (or restart) time?

    elif clawdata.output_style == 2:
        # Specify a list of output times.
        clawdata.output_times = [0.5, 1.0]

    elif clawdata.output_style == 3:
        # Output every iout timesteps with a total of ntot time steps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 1
        clawdata.output_t0 = True

    clawdata.output_format = 'binary'      # 'ascii' or 'binary'
    clawdata.output_q_components = 'all'   # could be list such as [True,True]
    clawdata.output_aux_components = 'all'
    clawdata.output_aux_onlyonce = False    # output aux arrays only at t0

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 0

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==1: variable time steps used based on cfl_desired,
    # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # If dt_variable==0 then dt=dt_initial for all steps:
    clawdata.dt_initial = 0.016

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used, and max to allow without
    # retaking step with a smaller dt:
    clawdata.cfl_desired = 0.75
    clawdata.cfl_max = 1.0

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 500000

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2

    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'

    # For unsplit method, transverse_waves can be
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3

    # List of limiters to use for each wave family:
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = ['mc', 'mc', 'mc']

    clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms

    # Source terms splitting:
    #   src_split == 0 or 'none'
    #      ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov'
    #      ==> Godunov (1st order) splitting used,
    #   src_split == 2 or 'strang'
    #      ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'

    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 => user specified (must modify bcN.f to use this option)
    #   1 => extrapolation (non-reflecting outflow)
    #   2 => periodic (must specify this at both boundaries)
    #   3 => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = 'extrap'
    clawdata.bc_upper[0] = 'extrap'

    clawdata.bc_lower[1] = 'extrap'
    clawdata.bc_upper[1] = 'extrap'

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif np.abs(clawdata.checkpt_style) == 1:
        # Checkpoint only at tfinal.
        pass

    elif np.abs(clawdata.checkpt_style) == 2:
        # Specify a list of checkpoint times.
        clawdata.checkpt_times = [0.1, 0.15]

    elif np.abs(clawdata.checkpt_style) == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:
    # ---------------
    amrdata = rundata.amrdata

    # max number of refinement levels:
    amrdata.amr_levels_max = 6
    # amrdata.amr_levels_max = 7

    # List of refinement ratios at each level (length at least mxnest-1)
    amrdata.refinement_ratios_x = [2, 2, 2, 2, 2, 4] # 200 m
    amrdata.refinement_ratios_y = [2, 2, 2, 2, 2, 4]
    amrdata.refinement_ratios_t = [2, 2, 2, 2, 2, 4]
    # amrdata.refinement_ratios_x = [2, 2, 2, 2, 4, 4] # 100 m
    # amrdata.refinement_ratios_y = [2, 2, 2, 2, 4, 4]
    # amrdata.refinement_ratios_t = [2, 2, 2, 2, 4, 4]
    # amrdata.refinement_ratios_x = [2, 2, 2, 2, 4, 8] # 50 m
    # amrdata.refinement_ratios_y = [2, 2, 2, 2, 4, 8]
    # amrdata.refinement_ratios_t = [2, 2, 2, 2, 4, 8]

    # 1 / (4*2*2*2*2*4*8) degrees
    # Note: 1 degree = 10 km

    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length maux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).

    amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center',
                        'center', 'center']

    # Flag using refinement routine flag2refine rather than richardson error
    amrdata.flag_richardson = False    # use Richardson?
    amrdata.flag2refine = True

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 3

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width = 2

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.700000

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0

    #  ----- For developers -----
    # Toggle debugging print statements:
    amrdata.dprint = False      # print domain flags
    amrdata.eprint = False      # print err est flags
    amrdata.edebug = False      # even more err est flags
    amrdata.gprint = False      # grid bisection/clustering
    amrdata.nprint = False      # proper nesting output
    amrdata.pprint = False      # proj. of tagged points
    amrdata.rprint = False      # print regridding summary
    amrdata.sprint = False      # space/memory output
    amrdata.tprint = False      # time step reporting each level
    amrdata.uprint = False      # update/upbnd reporting

    # More AMR parameters can be set -- see the defaults in pyclaw/data.py

    # == setregions.data values ==
    regions = rundata.regiondata.regions
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]
    # Entire region
    regions.append([1,5,
                    rundata.clawdata.t0,rundata.clawdata.tfinal,
                    rundata.clawdata.lower[0], rundata.clawdata.upper[0],
                    rundata.clawdata.lower[1], rundata.clawdata.upper[1]])

    # Caribbean region (NW)
    regions.append([1,rundata.amrdata.amr_levels_max,
                    rundata.clawdata.t0,rundata.clawdata.tfinal,
                    16,28,-86,-68])

    # Caribbean region (SE)
    regions.append([1,rundata.amrdata.amr_levels_max,
                    rundata.clawdata.t0,rundata.clawdata.tfinal,
                    9,21,-69,-58])

    # == Gauges == *
    gauges = rundata.gaugedata.gauges
    
    # Gauges from PLSMSL Stations https://www.psmsl.org/products/gloss/glossmap.html
    # 203: Port of Spain Trinidad and Tobago, 1, -61.517, 10.650, 
    gauges.append([1, -61.517, 10.650,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Gauges from Sea Level Station Monitoring Facility
    # Prickley Bay, Grenada Station, 2, -61.764828, 12.005392
    gauges.append([2, -61.764828, 12.005392,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Calliaqua Coast Guard Base, Saint Vincent & Grenadines, 3, -61.1955, 13.129912
    gauges.append([3, -61.1955, 13.129912,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Ganter's Bay, Saint Lucia, 4,-60.997351, 14.016428
    gauges.append([4, -60.997351, 14.016428,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Fort-de-France, Martinique2, 5, 
    gauges.append([5, -61.063333, 14.601667,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Roseau Dominica, 6, 61.3891833, 15.31385
    gauges.append([6, -61.3891833, 15.31385,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Pointe a Pitre, Guadeloupe, 7, -61.531452, 16.224398
    gauges.append([7, -61.531452, 16.224398,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Parham (Camp Blizard), Antigua, 8, -61.7833, 17.15
    gauges.append([8, -61.7833, 17.15,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Blowing Point, Anguilla, 9, -63.0926167, 18.1710861
    gauges.append([9, -63.0926167, 18.1710861,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Saint Croix, VI, 10, -64.69833, 17.74666
    gauges.append([10, -64.69833, 17.74666,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # San Juan, PR, 11, -66.1167, 18.4617
    gauges.append([11, -66.1167, 18.4617,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Barahona, Dominican Republic, 12, -71.092154, 18.208137
    gauges.append([12, -71.092154, 18.208137,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # George Town, Cayman Islands, 13, -81.383484, 19.295065
    gauges.append([13, -81.383484, 19.295065,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Settlement pt, Bahamas, 14, -78.98333, 26.6833324
    gauges.append([14, -78.98333, 26.6833324,
                                     rundata.clawdata.t0,
                                     rundata.clawdata.tfinal])
    # Force the gauges to also record the wind and pressure fields
    rundata.gaugedata.aux_out_fields = [4, 5, 6]

    # Add gauge support
    dx = 1/3600
    for g in gauges:
        t1 = g[3]
        t2 = g[4]
        x = g[1]
        y = g[2]
        regions.append([amrdata.amr_levels_max, amrdata.amr_levels_max,
                        t1, t2, x-dx, x+dx, y-dx, y+dx])
            


    # == fgmax_grids.data values ==

    from clawpack.geoclaw import fgmax_tools
    # set num_fgmax_val = 1 to save only max depth,
    #                     2 to also save max speed,
    #                     5 to also save max hs,hss,hmin
    rundata.fgmax_data.num_fgmax_val = 1  # Save depth only
    # rundata.fgmax_data.num_fgmax_val = 2  # Save depth and speed

    fgmax_grids = rundata.fgmax_data.fgmax_grids  # empty list to start

    # Now append to this list objects of class fgmax_tools.FGmaxGrid
    # specifying any fgmax grids.
    # Note: 1 arcsec = 30 m
    # Make sure x1 < x2, y1 < y2
    # Copy regions file for future 'make plots'
    regions_file = 'regions.json'
    regions_file_output = f'_output/{regions_file}'
    os.system(f'cp ../{regions_file} {regions_file_output}')
    import json
    with open(regions_file_output,"r") as file:
        fgmax_regions = json.load(file)

    for fr in fgmax_regions:
    # Points on a uniform 2d grid:
        fg = fgmax_tools.FGmaxGrid()
        fg.point_style = 2  # uniform rectangular x-y grid  
        fg.x1 = fr['x1']
        fg.x2 = fr['x2']
        fg.y1 = fr['y1']
        fg.y2 = fr['y2']
        # desired resolution of fgmax grid
        if 'dx' in fr.keys():
            fg.dx = fr['dx']
        else:
            # fg.dx = 5 / 3600.  
            fg.dx = 10 / 3600.  
        fg.min_level_check = 3 # which levels to monitor max on
        fg.tstart_max = clawdata.t0  # just before wave arrives
        fg.tend_max = clawdata.tfinal    # when to stop monitoring max values
        fg.dt_check = 60.      # how often to update max values
        fg.interp_method = 0   # 0 ==> pw const in cells, recommended
        fgmax_grids.append(fg)  # written to fgmax_grids.data

    return rundata
    # end of function setrun
    # ----------------------


# -------------------
def generate_geo_data(rundata):
    """
    Set GeoClaw specific runtime parameters.
    For documentation see ....
    """

    geo_data = rundata.geo_data

    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.earth_radius = 6367.5e3
    geo_data.rho = 1025.0
    geo_data.rho_air = 1.15
    geo_data.ambient_pressure = 101.3e3

    # == Forcing Options
    geo_data.coriolis_forcing = True
    geo_data.friction_forcing = True
    geo_data.friction_depth = 1e10

    # == Algorithm and Initial Conditions ==
    # Due to seasonal swelling of gulf we set sea level higher
    geo_data.sea_level = 0
    geo_data.dry_tolerance = 1.e-2

    # Refinement Criteria
    refine_data = rundata.refinement_data
    refine_data.wave_tolerance = 1.0
    refine_data.speed_tolerance = [1.0, 2.0, 3.0, 4.0]
    refine_data.deep_depth = 300.0
    refine_data.max_level_deep = 4
    refine_data.variable_dt_refinement_ratios = True

    # == settopo.data values ==
    topo_data = rundata.topo_data
    topo_data.topofiles = []
    topo_data.topo_missing = -32768
    # for topography, append lines of the form
    #   [topotype, fname]
    # See regions for control over these regions, need better bathy data for
    # the smaller domains

    # Entire domain
    minlon = rundata.clawdata.lower[0]
    maxlon = rundata.clawdata.upper[0]
    minlat = rundata.clawdata.lower[1]
    maxlat = rundata.clawdata.upper[1]
    # ERDDAP Data Access Form call
    # ETOPO - 1 arcminute resolution
    topo_filename = f'etopo180.esriAscii?altitude[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]'
    topo_url = 'http://coastwatch.pfeg.noaa.gov/erddap/griddap/' + topo_filename
    clawutil.data.get_remote_file(topo_url)
    topo_path = os.path.join(scratch_dir, topo_filename)
    topo_data.topofiles.append([3, topo_path])
    # Caribbean region
    # minlat = 9
    # maxlat = 28
    # minlon = -86
    # maxlon = -58
    # Break up into two regions

    # Caribbean region (NW)
    minlat = 16
    maxlat = 28
    minlon = -86
    maxlon = -68
    # ERDDAP Data Access Form call
    # SRTM15 - 15 arcsecond resolution
    # topo_filename = f'srtm15plus.esriAscii?z[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]'
    # GEBCO_2020 Grid - 15 arcsecond resolution
    topo_filename = f'GEBCO_2020.esriAscii?elevation[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]'
    topo_url = 'http://coastwatch.pfeg.noaa.gov/erddap/griddap/' + topo_filename 
    clawutil.data.get_remote_file(topo_url)
    topo_path = os.path.join(scratch_dir, topo_filename)
    topo_data.topofiles.append([3, topo_path])
    # Caribbean region (SE)
    minlat = 9
    maxlat = 21
    minlon = -69
    maxlon = -58
    # ERDDAP Data Access Form call
    # SRTM15 - 15 arcsecond resolution
    # topo_filename = f'srtm15plus.esriAscii?z[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]'
    # GEBCO_2020 Grid - 15 arcsecond resolution
    topo_filename = f'GEBCO_2020.esriAscii?elevation[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]'
    topo_url = 'http://coastwatch.pfeg.noaa.gov/erddap/griddap/' + topo_filename 
    clawutil.data.get_remote_file(topo_url)
    topo_path = os.path.join(scratch_dir, topo_filename)
    topo_data.topofiles.append([3, topo_path])
    # Virgin Islands High resolution 
    # # High resolution (1 arcsec / 30m) Virgin Islands topo-bathy (roughly 2^-11 deg)
    # VI_highres_filename = 'usvi_1_mhw_2014.nc'
    # VI_highres_url = 'https://www.ngdc.noaa.gov/thredds/ncss/regional/' + VI_highres_filename
    # clawutil.data.get_remote_file(VI_highres_url)
    # # Note: for faster performance, crop this file to 17.5N to 18.5N x -65.1W to -64.3W
    # # gdal_translate -projwin -65.1 18.5 -64.3 17.5 usvi_1_mhw_2014.nc usvi_1_mhw_2014_crop.nc
    # # Note 2: for even faster performance, subsample to 60m resolution:
    # # gdalwarp -tr 0.00056 0.00056 -r cubic usvi_1_mhw_2014_crop.nc usvi_1_mhw_2014_crop_subsample.nc
    # # Then update the filename:
    # # VI_highres_filename = 'usvi_1_mhw_2014_crop_subsample.nc'
    # topo_path = os.path.join(scratch_dir, VI_highres_filename)
    # topo_data.topofiles.append([4, topo_path])

    # == fgout grids ==
    # new style as of v5.9.0 (old rundata.fixed_grid_data is deprecated)
    # set rundata.fgout_data.fgout_grids to be a 
    # list of objects of class clawpack.geoclaw.fgout_tools.FGoutGrid:
    #rundata.fgout_data.fgout_grids = []

    # ================
    #  Set Surge Data
    # ================
    data = rundata.surge_data

    # Source term controls
    data.wind_forcing = True
    data.drag_law = 1
    data.pressure_forcing = True

    data.display_landfall_time = True

    # AMR parameters, m/s and m respectively
    data.wind_refine = [20.0, 40.0, 60.0]
    data.R_refine = [60e3, 40e3, 20e3]

    # Storm parameters - Parameterized storm (Holland 1980)
    data.storm_specification_type = 'holland80'  # (type 1)

    # =======================
    #  Set Variable Friction
    # =======================
    data = rundata.friction_data

    # Variable friction
    data.variable_friction = True

    # Region based friction
    # Entire domain
    data.friction_regions.append([rundata.clawdata.lower,
                                  rundata.clawdata.upper,
                                  [np.infty, 0.0, -np.infty],
                                  [0.030, 0.022]])

    return rundata
    # ----------------------
