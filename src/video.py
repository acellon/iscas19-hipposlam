# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from brian2 import *
import vco
from matplotlib import animation

# %matplotlib notebook

# %%
N = 12
numrow = 6; numcol = 12;
rho_num = 0.14
rhos = rho_num*2 * (np.sqrt(3) ** np.arange(numrow))
thetas = np.pi + 2.0*np.pi*(np.arange(numcol))/numcol


VCOmatrix = [[vco.VCO_model(N, rhos[i], thetas[j]) for j in range(numcol)] for i in range(numrow)]

# %%
# Create a simulated running path (constant velocity, running around in a circle)
d = np.arange(0, 2 * np.pi, 0.001)
v = 2*np.ones_like(d)
pol_path = np.asarray([v,d]).T
x = v * np.cos(d)
y = v * np.sin(d)


# %%
# Create tuning curves by calculating angular frequency
tune = np.zeros((d.shape[0],N))
for cell in range(N):
    tune[:,cell] = VCOmatrix[0][0].get_angular_freq(cell, pol_path)

# %%
# Plot the tuning curves
fig, axs = plt.subplots(3,4,sharex=True,sharey=True,figsize=(8,6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for cell in range(N):
    axs[cell].plot(d,tune[:,cell])
    axs[cell].set_title('Cell %d' % cell)
    axs[cell].grid()
    
    if not (cell) % 4:
        axs[cell].set_ylabel('Frequency (Hz)')
    if cell >= 8:
        axs[cell].set_xlabel('Direction')
        axs[cell].set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
        axs[cell].set_xticklabels(["$0$", "$\pi/2$", "$\pi$","$3\pi/2$", "$2 \pi$"]);

plt.tight_layout()
plt.suptitle(r"VCO Tuning Curves ($\rho = 0.14, \theta = \pi$)",fontsize=16,y=1.02);
plt.show()

# %%
arena_size = 5

# %%
# border cell (Fig. 7)
weights_border = np.full([6,12],np.nan)
weights_border[:,9] = [2,3,5,7,11,9]

# large grid (Fig. 7)
weights_lgrid = np.full([6,12],np.nan)
weights_lgrid[3,0] = 9
weights_lgrid[3,4] = 9
weights_lgrid[3,8] = 3

# small grid (Fig. 7)
weights_sgrid = np.full([6,12],np.nan)
weights_sgrid[3,1] = 1
weights_sgrid[3,5] = 9
weights_sgrid[3,9] = 3

# place cell (Fig. 7)
weights_place = np.full([6,12],np.nan)
rot_place = 8*np.pi/6. #orientation of the tuning function is zero by default
weights_place[2,:] = [11,11,0,0,0,0,0,0,11,10,10,10]
weights_place[3,:] = [10,11,0,0,0,0,0,0,10,10,10,10]
weights_place[4,:] = [10,11,0,1,2,1,0,0,10, 9, 8, 9]

# curved border (Fig. 9)
weights_cborder = np.full([6,12],np.nan)
for col in [0,1,2,3,4,11]:
    weights_cborder[:,col] = [1,1,2,4,7,1]
    
#lumpy border (supplemental)
weights_lborder = np.full([6,12],np.nan)
weights_lborder[:,0] = [1,2,3,5,8,3]
weights_lborder[2,2] = 6
weights_lborder[3,2] = 1

# multi-field dentate place cell in square box (supplemental)
weights_dplace = np.full([6,12],np.nan)
weights_dplace[2,:] = [8,0,1,1,7,4,2,7,8,8,4,2]
weights_dplace[3,:] = [3,0,3,0,5,0,1,4,5,9,9,11]
weights_dplace[4,:] = [0,2,2,10,7,0,0,5,6,2,6,4]

# %%
border_norm, border_env = vco.matrix_sum(VCOmatrix, weights_border,  arena_size)
lgrid_norm, lgrid_env   = vco.matrix_sum(VCOmatrix, weights_lgrid,   arena_size)
sgrid_norm, sgrid_env   = vco.matrix_sum(VCOmatrix, weights_sgrid,   arena_size)
place_norm, place_env   = vco.matrix_sum(VCOmatrix, weights_place,   arena_size)
cbord_norm, cbord_env   = vco.matrix_sum(VCOmatrix, weights_cborder, arena_size)
lbord_norm, lbord_env   = vco.matrix_sum(VCOmatrix, weights_lborder, arena_size)
dplace_norm, dplace_env = vco.matrix_sum(VCOmatrix, weights_dplace,  arena_size)


# %%
fig, axs = plt.subplots(7,2,sharex=True,sharey=True,figsize=(6,14))

axs[0][0].imshow(lgrid_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[0][0].set_title('Large grid: Envelope')
axs[0][1].imshow(lgrid_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[0][1].set_title('Large grid: Normalized Envelope')

axs[1][0].imshow(sgrid_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[1][0].set_title('Small grid: Envelope')
axs[1][1].imshow(sgrid_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[1][1].set_title('Small grid: Normalized Envelope')

axs[2][0].imshow(border_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[2][0].set_title('Border: Envelope')
axs[2][1].imshow(border_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[2][1].set_title('Border: Normalized Envelope')

axs[3][0].imshow(cbord_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[3][0].set_title('Curved border: Envelope')
axs[3][1].imshow(cbord_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[3][1].set_title('Curved border: Normalized Envelope')

axs[4][0].imshow(lbord_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[4][0].set_title('Lumpy border: Envelope')
axs[4][1].imshow(lbord_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[4][1].set_title('Lumpy border: Normalized Envelope')

axs[5][0].imshow(place_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[5][0].set_title('Place: Envelope')
axs[5][1].imshow(place_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[5][1].set_title('Place: Normalized Envelope')

axs[6][0].imshow(dplace_env, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[6][0].set_title('Dentate place: Envelope')
axs[6][1].imshow(dplace_norm, aspect='auto', cmap='jet', extent=(-arena_size,arena_size,-arena_size,arena_size))
axs[6][1].set_title('Dentate place: Normalized Envelope')

plt.tight_layout()


# %%
arena_size = 2
N=20000
path = vco.randwalk(0.1,N,2*arena_size)

# %%
path

# %%
#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_lgrid, path))
#plt.title('Large Grid: Random Walk');

#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_sgrid, path))
#plt.title('Small Grid: Random Walk');

#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_border, path))
#plt.title('Border: Random Walk');

#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_cborder, path))
#plt.title('Curved border: Random Walk');

#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_lborder, path))
#plt.title('Lumpy border: Random Walk');

vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_place, path))
plt.title('Place: Random Walk');

#vco.plot_randwalk(path, vco.matrix_sum_rw(VCOmatrix, weights_dplace, path))
#plt.title('Place: Random Walk');


# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

envelope = abs(vco.matrix_sum_rw(VCOmatrix, weights_sgrid, path))
max_env = np.max(envelope)
env_thresh = envelope - 0.65*max_env
env_thresh[env_thresh<0] = 0
max_env  = np.max(env_thresh)
env_norm = env_thresh/max_env

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-arena_size, arena_size), ylim=(-arena_size, arena_size))
plt.xticks([])
plt.yticks([])

line = LineCollection([], cmap=plt.get_cmap('jet'))
line.set_array(env_norm)
ax.add_collection(line)

# initialization function: plot the background of each frame
def init():
    line.set_segments([])
    return line,

# animation function.  This is called sequentially
def animate(i, path):
    x = path[0:i,0]
    y = path[0:i,1]
    points = np.array([x,y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    line.set_segments(segs)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=[path],
                           frames=20000, interval=5)

anim.save('bigPlace.mp4', fps=300, extra_args=['-vcodec', 'libx264'])

# %%
x = path[:,0]
y = path[:,1]

# %%
a = VCOmatrix[3][0].get_envelope(9,x,y)
b = VCOmatrix[3][4].get_envelope(9,x,y)
c = VCOmatrix[3][8].get_envelope(3,x,y)
timeddt = 1*second

# %%
offset = 4
envs = [[offset+a], [offset+b], [offset+c]]*Hz
envs = np.reshape(envs,[3,10000]).T

ta = TimedArray(a*Hz, dt=timeddt)
tb = TimedArray(b*Hz, dt=timeddt)
tc = TimedArray(c*Hz, dt=timeddt)
tf = TimedArray(envs, dt=timeddt)

# %%
neuron_eq = '''
        dVm/dt = (glm / Cm) * (Vm_r - Vm) : volt

        glm = flm * Cl                    : siemens
    '''
reset_eq = '''
        Vm = Vm_r
    '''
presyn_eq = '''
        Vm_old = Vm
        Vm = Vm_old + Vsyn
    '''

# Synapse equation is the same for both modes!
syn_eq = '''
    Vsyn = (W/Cm)*(Em - Vm) : volt
    Em                      : volt
    W                       : farad
'''

# IFAT specific definitions
fF = 0.001 * pF
Vdd = 5 * volt
Cm = Ct = 440 * fF
Cl = 2 * fF

W_vals  = np.array([5, 10, 20, 40, 80]) * fF
Em_vals = np.array([0, 1/3, 2/3, 1]) * Vdd

par_ctrl = 1.0
par_leak_time = 12.5 * ms

# Model parameters
Vm_r = 1 * volt
flm  = 0.5 * kHz
Csm  = W_vals[0]

Vt_r = 2 * volt
flt  = 0 * MHz
Cst  = 0 * fF

# %%
start_scope()

Ptest = PoissonGroup(1, rates='ta(t)+tb(t)+tc(t)+3*Hz')
#G = NeuronGroup(1, neuron_eq, threshold='Vm>Vt_r', reset=reset_eq)
#Psyn = Synapses(Ptest, G, syn_eq, on_pre=presyn_eq)
#Psyn.connect()
#Psyn.Em = Em_vals[3]
#Psyn.W = W_vals[2] + W_vals[0]

#e_spmon = SpikeMonitor(G)
#e_vmon = StateMonitor(G, 'Vm', record=True)
i_spmon = SpikeMonitor(Ptest)
inrate = PopulationRateMonitor(Ptest)
ratecheck = StateMonitor(Ptest, 'rates',record=True)

store()

# %%
run(len(a)*timeddt/1)

# %%
for i in range(Ptest.N):
    plt.plot(ratecheck.t/second, ratecheck.rates[i]/Hz)

plt.show()

# %%
plt.plot(i_spmon.t/second, i_spmon.i+7,'.')
plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

