import numpy as np
from simulation import *
import matplotlib.pyplot as plt # temporary visualization for testing
import matplotlib.animation as animation
# import functions for animation as well

H = 0.01 # step size

x = np.zeros((3,2)) # three bodies, two components: position and velocity
y = np.zeros((3,2))
m = np.ones(3) # masses of the three bodies

# set initial conditions
x[0] = np.array([0,0])
y[0] = np.array([0,0])
m *= 1e5
m[0] = 1e10
for i in range(1,3):
    x[i] = np.random.normal(size=(2,))
    y[i] = np.random.normal(size=(2,))
    dist = np.linalg.norm([x[i]-x[0], y[i]-y[0]])
    vi = np.sqrt(G*m[0]/dist)
    x[i][1] = vi*(y[i][0]-y[0][0])/dist
    y[i][1] = -vi*(x[i][0]-x[0][0])/dist

SIM_LEN = 500
simulation = []
for i in range(SIM_LEN):
    x, y = simulate_step(x, y, m, H)
    simulation.append([np.copy(x[:,0]), np.copy(y[:,0])])

max_coord = np.max(np.abs(np.concatenate(simulation)))
fig = plt.figure()
scatter = plt.scatter(x[:,0], y[:,0])
ani = animation.FuncAnimation(fig, lambda i: scatter.set_offsets(np.array([simulation[i][0], simulation[i][1]]).T),
                              frames=range(SIM_LEN), interval=20)
fig.get_axes()[0].set_xlim(-max_coord, max_coord)
fig.get_axes()[0].set_ylim(-max_coord, max_coord)

plt.show()