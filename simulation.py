import numpy as np
EPS = 1e-12 # epsilon for numerical stability

def runge_kutta_step(f, y, h):
    # assuming f does not depend on t (time symmetry)
    k1 = f(y)
    k2 = f(y + h/2*k1)
    k3 = f(y + h/2*k2)
    k4 = f(y + h*k3)
    return h/6*(k1 + 2*k2 + 2*k3 + k4)

G = 6.674e-11 * 1e3 * 1e-9 # gravitational constant in km^3 Mg^-1 s^-2
def gravity(x1, x2, y1, y2, m2):
    # acceleration due to gravity of m2 on m1
    return -G*m2*(x1-x2)/(np.linalg.norm((x1-x2, y1-y2))**3 + EPS)

def simulate_step(xy, m, h):
    # xy is an array of shape (N,4) for N bodies and four components:
    # x, y, vx, vy
    # m is an array of shape (N) for the masses of N bodies
    N = xy.shape[0]
    new_xy = np.copy(xy)
    for i in range(N):
        for j in range(N):
            if i == j:
                f_x = lambda xyi: np.array([xyi[2], xyi[3], 0, 0])
                step = runge_kutta_step(f_x, xy[i], h)
                new_xy[i][0] += step[0]
                new_xy[i][1] += step[1]
                continue
            f_x = lambda xyi: np.array([xyi[2], xyi[3],
                                       gravity(xyi[0], xy[j,0], xyi[1], xy[j,1], m[j]),
                                       gravity(xyi[1], xy[j,1], xyi[0], xy[j,0], m[j])])
            step = runge_kutta_step(f_x, xy[i], h)
            new_xy[i][2] += step[2]
            new_xy[i][3] += step[3]
    return new_xy