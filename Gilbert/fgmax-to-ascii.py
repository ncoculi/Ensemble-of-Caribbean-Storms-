def fgmax_to_ascii(fgno=1, nodata_value=-9999):
    from clawpack.geoclaw import fgmax_tools
    fg = fgmax_tools.FGmaxGrid()
    fg.read_fgmax_grids_data(fgno)
    fg.read_output(outdir='_output')
    fg.B0 = fg.B  # no seafloor deformation in this problem
    
    import numpy as np
    fg.h_onshore = np.ma.masked_where(fg.B0 < 0., fg.h)
    # Mask less than 0.1 m
    cutoff = 0.1
    nodata_value = -9999
    inundation = np.ma.masked_where(fg.h_onshore < cutoff, fg.h_onshore).filled(nodata_value).transpose()
    from clawpack.geoclaw import topotools
    topo = topotools.Topography()
    topo.x = fg.X[:,0]
    topo.y = fg.Y[0,:]
    topo.Z = inundation
    dy = topo.y[1] - topo.y[0]
    dx = topo.x[1] - topo.x[0]
    topo._delta = (dx,dy)
    topo.Z = inundation
    topo.write(f'_output/fgmax{fgno:04}.asc', header_style='asc', fill_value=nodata_value, topo_type=3)
    
for i in range(8):
    j = i+1
    try:
        fgmax_to_ascii(fgno=j)
    except:
        print('Skipping fgno={j}')
        pass