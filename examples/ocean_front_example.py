import sys; sys.path.append("..")

import time, logging
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import pi
from dedalus.extras import flow_tools
import scipy.integrate as integrate

logger = logging.getLogger(__name__)

import dedaLES

# Some convenient constants
second = 1.0
minute = 60*second
hour   = 60*minute
day    = 24*hour
siderealyear = 365*day + 6*hour + 9*minute + 9.76*second
omega = 2*pi / siderealyear

# Physical constants
α  = 1.19e-4                # Thermal expansion coefficient for water at 10ᵒC [K⁻¹]
g  = 9.81                   # Graviational acceleration [m s⁻²]
ρ0 = 1027.62                # Reference density [kg m⁻³]
cP = 4003.0                 # Specific heat of seawater at 10ᵒC [J kg⁻¹ K⁻¹]
κ  = 1.0e-6                # Thermal diffusivity of seawater [m² s⁻¹]
ν  = 1.0e-6                # Viscosity of seawater [m² s⁻¹]

# Denbo and Skyllinstad (1995) experiment parameters
Q = -10.0                     # Surface cooling rate [W m⁻²]
bz_deep = 2e-6              # Deep buoyancy gradient
h0 = 60.0                  # Initial mixed layer depth
d = 1                   # Transition thickness from deep stratified to mixed layers
f = 1.0e-4                  # Coriolis parameters [s⁻¹]
Msq = 3.0e-8               # Zonal buoyancy gradient [s^-2]

nx = 256                   # Horizontal resolution
ny = 256
nz = 48                     # Vertical resolution
Lx = 500                    # Domain across-front extent [m]
Ly = 500                     # Domain along-front extent
Lz = 140                   # Domain vertical extent [m]
a = 1e-7                    # Non-dimensional initial noise amplitude


# Calculated parameters
bz_surf = Q*α*g / (cP*ρ0*κ) # Unstable surface buoyancy gradient [s⁻²]
b_init = (Lz-h0)*bz_deep    # Initial buoyancy at bottom of domain
b_noise = a*b_init          # Noise amplitude for initial buoyancy condition
b_noise = a

# Construct model
closure = dedaLES.ConstantSmagorinsky()
#closure = dedaLES.AnisotropicMinimumDissipation()
model = dedaLES.FrontalZone(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz, ν=ν, κ=κ, zbottom=-Lz, bz_deep=bz_deep, bz_surf=bz_surf,  closure=closure, H=Lz, Msq=Msq)
Δz_min = np.min(model.problem.domain.grid_spacing(2))
# Boundary conditions
model.set_bc("no penetration", "top", "bottom")
#model.set_bc("free slip", "top", "bottom")
#model.set_bc("freeslip", "top", "bottom")
model.set_tracer_gradient_bc('u', "top", gradient="0")
model.set_tracer_gradient_bc('u', "bottom", gradient="0")
model.set_tracer_gradient_bc('v', "top", gradient="-Vgz")
model.set_tracer_gradient_bc('v', "bottom", gradient="-Vgz")
model.set_tracer_gradient_bc("b", "top", gradient="bz_surf")
model.set_tracer_gradient_bc("b", "bottom", gradient='bz_deep')

model.build_solver(timestepper='CNAB2')

# Initial condition
def smoothstep(z, d): 
    return 0.5*(1 + np.tanh(z/d))

b0 = bz_deep * (model.z + h0) * smoothstep(-model.z-h0, d)

# Add noise ... ?
noise = dedaLES.random_noise(model.domain)
b0 = b0 + b_noise * noise
model.set_fields(b=b0, u=0, v=0, w=0)

# CFL conditions
CFL = flow_tools.CFL(model.solver, initial_dt = 1e-2, cadence = 1, max_change = 1.5, max_dt = 10, safety=0.5)
CFL.add_velocities(('u', 'v+Vg', 'w'))

#Analysis tasks
snap = model.solver.evaluator.add_file_handler('snapshots', sim_dt=30)
#snap.add_task("interp(b, z=0)", scales=1, name='b surface')
#snap.add_task("interp(u, z=0)", scales=1, name='u surface')
#snap.add_task("interp(v, z=0)", scales=1, name='v surface')

snap.add_task("interp(b, z=-25)", scales=1, name='b 25')
snap.add_task("interp(u, z=-25)", scales=1, name='u 25')
snap.add_task("interp(v, z=-25)", scales=1, name='v 25')
snap.add_task("interp(w, z=-25)", scales=1, name='w 25')

snap.add_task("interp(b, y=0)", scales=1,  name='b mid')
snap.add_task("interp(u, y=0)", scales=1, name='u mid')
snap.add_task("interp(v, y=0)", scales=1, name='v mid')
snap.add_task("interp(w, y=0)", scales=1, name='w mid')
#snap.add_task("interp('ν_sgs', y=0)", scales=1, name='sg mid')

# Flow properties
flow = flow_tools.GlobalFlowProperty(model.solver, cadence=1)
flow.add_property("sqrt(u*u + v*v + w*w)*H / ν", name='Re_domain')
flow.add_property("sqrt(ε) / ν", name='Re_dissipation')
flow.add_property("ε", name="dissipation")
flow.add_property("ε_sgs", name="subgrid_dissipation")
flow.add_property("w*w", name="w_square")
flow.add_property('ν_sgs', name='subgrid_visc')

model.stop_at(sim_time = 24*3600*5)

for i in range(0, 9):
    logger.info(model.problem.boundary_conditions[i]['raw_equation'])

# Main loop
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    dt = CFL.compute_dt()
    while model.solver.ok: 
        model.solver.step(dt)#this ordering is necessary so that the model 'sees' subgrid terms...hacky
        dt = CFL.compute_dt()
        dt = min(dt, 0.5*0.5*Δz_min**2/flow.max("subgrid_visc")) # note should also check diffusion

        if model.time_to_log(1): 
            logger.info('Iter: {}, Time: {:.2f}, dt: {:.4f}'.format(model.solver.iteration, model.solver.sim_time/hour, dt))
           # logger.info("     Max domain Re = {:.6f}".format(flow.max("Re_domain")))
           # logger.info("Max dissipation Re = {:.6f}".format(flow.max("Re_dissipation")))
            logger.info("Max subgrid visc   = {:.6f}".format(flow.max("subgrid_visc")))
           # logger.info("   Average epsilon = {:.6f}".format(flow.volume_average("dissipation")))
           # logger.info("Average SG epsilon = {:.6f}".format(flow.volume_average("subgrid_dissipation")))
            logger.info("     Average rms w = {:.6f}".format(np.sqrt(flow.volume_average("w_square"))))
            logger.info("     Max w         = {:.6f}".format(np.sqrt(flow.max("w_square"))))            
        #if model.time_to_log(100):
        #    btemp = model.solver.state['b']['g']
        #    wtemp = model.solver.state['w']['g']
        #    plt.figure()
        #    plt.pcolor(wtemp[:,:,-5])
        #    plt.savefig('/data/thomas/jacob13/LESSandbox/wtemp.png')
        #    plt.figure()
        #    plt.pcolor(btemp[:,:,-5])
        #    plt.savefig('/data/thomas/jacob13/LESSandbox/btemp.png')
 


except:
    logger.error('Exception raised, triggering end of main loop.')
    raise

finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %model.solver.iteration)
    logger.info('Sim end time: %f' %model.solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time) / hour * model.domain.dist.comm_cart.size))
        










































