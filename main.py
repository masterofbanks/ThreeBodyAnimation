import numpy as np
from simulation import *
import matplotlib.pyplot as plt # temporary visualization for testing
import matplotlib.animation as animation
# import functions for animation as well

H = 86400//24 # step size in seconds (equal to 2 hours)
N = 2 # number of bodies
SIM_LEN = 100000
SIM_SPEED = 50

x = np.zeros((N,2)) # N bodies, two components: position and velocity
y = np.zeros((N,2))
m = np.ones(N) # masses of the N bodies

# set initial conditions
x[0] = np.array([0,0])
y[0] = np.array([0,0])
m = np.ones(N) * 1e24
m[0] = 1e30
for i in range(1,N):
    x[i][0] = 1e8 * (i*10-9)
    y[i][0] = x[i][0]
    dist = np.linalg.norm((x[i][0]-x[0][0], y[i][0]-y[0][0]))
    vi = np.sqrt(G*m[0]/dist)
    x[i][1] = vi*(y[i][0]-y[0][0])/dist
    y[i][1] = -vi*(x[i][0]-x[0][0])/dist

simulation = []
for i in range(SIM_LEN):
    x, y = simulate_step(x, y, m, H)
    sim = np.array([x[:,0], y[:,0]]).T
    simulation.append(sim)
simulation = np.array(simulation)

TRAIL_LEN = 10000
max_coord = np.max(np.abs(np.concatenate(simulation))) * 1.2
fig = plt.figure()
scatter = plt.scatter(x[:,0], y[:,0], s=np.log10(m))
trail_lines = []
for i in range(N):
    trail_lines.append(plt.plot([], [], color=scatter.get_facecolor()[0])[0])
def animate_func(i):
    scatter.set_offsets(simulation[i*SIM_SPEED])
    for j in range(N):
        trail_lines[j].set_data(simulation[max(0,(i-TRAIL_LEN)*SIM_SPEED):i*SIM_SPEED+1,j,0],
                                simulation[max(0,(i-TRAIL_LEN)*SIM_SPEED):i*SIM_SPEED+1,j,1])
    return scatter, trail_lines
ani = animation.FuncAnimation(fig, animate_func, frames=range(SIM_LEN//SIM_SPEED), interval=20)
fig.get_axes()[0].set_xlim(-max_coord, max_coord)
fig.get_axes()[0].set_ylim(-max_coord, max_coord)

plt.gca().set_aspect('equal')
plt.show()