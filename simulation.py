import numpy as np
EPS = 1e-7 # epsilon for numerical stability

def runge_kutta_step(f, y, h):
    # assuming f does not depend on t (time symmetry)
    k1 = f(y)
    k2 = f(y + h/2*k1)
    k3 = f(y + h/2*k2)
    k4 = f(y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

G = 6.67408e-11 # gravitational constant
def gravity(x1, x2, y1, y2, m2):
    # acceleration due to gravity of m2 on m1
    return -G*m2*(x1-x2)/(np.linalg.norm((x1-x2, y1-y2))+EPS)**3

def simulate_step(x, y, m, h):
    # x, y are each arrays of shape (N,2) for N bodies and two components:
    # one position, one velocity.
    # m is an array of shape (N) for the masses of N bodies
    N = x.shape[0]
    new_x = np.copy(x)
    new_y = np.copy(y)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            f_x = lambda xi: np.array([xi[1], gravity(xi[0], x[j,0], y[i,0], y[j,0], m[j])])
            f_y = lambda yi: np.array([yi[1], gravity(yi[0], y[j,0], x[i,0], x[j,0], m[j])])
            new_x[i] = runge_kutta_step(f_x, new_x[i], h)
            new_y[i] = runge_kutta_step(f_y, new_y[i], h)
    return new_x, new_y