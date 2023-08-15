import numpy as np
from simulation import *
import matplotlib.pyplot as plt # temporary visualization for testing
import matplotlib.animation as animation
# import functions for animation as well

H = 86400//24 # step size in seconds (equal to 1 hour)
N = 4 # number of bodies
SIM_LEN = int(365.2422*24*1.5)
SIM_SPEED = 10

xy = np.zeros((N,4)) # N bodies, four components: x, y, vx, vy
m = np.ones(N) # masses of the N bodies

# set initial conditions
xy[0] = np.array([0,0,0,0])
m = np.ones(N)

PLANET_DISTANCES_AU = [
    0.39,
    0.72,
    1.00
]
MASSES_KG = [
    1.989e30, # sun
    3.285e23, # mercury
    4.867e24, # venus
    5.972e24, # earth
]
for i in range(N):
    m[i] = MASSES_KG[i] * 1e-3 # Mg

for i in range(1,N):
    dist = PLANET_DISTANCES_AU[i-1] * 1.496e8 # km
    angle = np.random.rand() * 2 * np.pi
    xy[i,0] = dist * np.cos(angle)
    xy[i,1] = dist * np.sin(angle)
    vi = np.sqrt(G*m[0]/dist)
    xy[i,2] = vi*(xy[i,1]-xy[0,1])/dist
    xy[i,3] = -vi*(xy[i,0]-xy[0,0])/dist

simulation = []
for i in range(SIM_LEN):
    xy = simulate_step(xy, m, H)
    sim = np.array([xy[:,0], xy[:,1]]).T
    simulation.append(sim)
simulation = np.array(simulation)

TRAIL_LEN = 50
max_coord = np.max(np.abs(np.concatenate(simulation))) * 1.2
fig = plt.figure()
scatter = plt.scatter(xy[:,0], xy[:,1], s=np.log(m/np.min(m)+1)*10)
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