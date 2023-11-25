import numpy as np
import scipy.integrate as integrate
EPS = 1e-12 # epsilon for numerical stability

G = 6.674e-11 * 1e3 * 1e-9 # gravitational constant in km^3 Mg^-1 s^-2
def gravity(x1, x2, y1, y2, m2):
    # acceleration due to gravity of m2 on m1
    return -G*m2*(x1-x2)/(np.linalg.norm((x1-x2, y1-y2))**3 + EPS)

def simulate_steps(xy, m, h, steps):
    # xy is an array of shape (N,4) for N bodies and four components:
    # x, y, vx, vy
    # m is an array of shape (N) for the masses of N bodies
    N = xy.shape[0]
    new_xy = np.copy(xy)

    def get_delta(i, yv):
        d = np.zeros(4)
        for j in range(N):
            if i == j:
                step = np.array([yv[2], yv[3], 0, 0])
                d[0] += step[0]
                d[1] += step[1]
                continue
            step = np.array([yv[2], yv[3],
                                    gravity(yv[0], new_xy[j,0], yv[1], new_xy[j,1], m[j]),
                                    gravity(yv[1], new_xy[j,1], yv[0], new_xy[j,0], m[j])])
            d[2] += step[2]
            d[3] += step[3]
        return d
    
    solver = []
    for i in range(N):
        s = integrate.RK45(
            lambda t,y: get_delta(i, y), 0, xy[i], t_bound=h*steps+1, max_step=h
        )
        solver.append(s)

    simulation = []
    for _ in range(steps):
        for i in range(N):
            solver[i].step()
            new_xy[i] = solver[i].y
        simulation.append(np.copy(new_xy))
    return np.array(simulation)
