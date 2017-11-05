import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy

dt = 0.1
time = np.arange(0, 20, dt)
LANDMARKS = np.array([[6., 4.]])

def measure(dt, x, Q, noise=False):
    z = np.zeros([2 * LANDMARKS.shape[0]])

    for i, m in enumerate(LANDMARKS):
        q = (m[0] - x[0]) ** 2 + (m[1] - x[1]) ** 2
        z[i * 2] = np.sqrt(q)
        z[i * 2 + 1] = np.arctan2(m[1] - x[1], m[0] - x[0]) - x[2]

    z = np.random.multivariate_normal(z, Q) if noise else z

    return z

def dfdx(xu, dt):
    v = u[0]
    w = u[1]

    return G

def dgdu(u, dt):
    v = u[0]
    w = u[1]

    theta = self.x[2]
    st = np.sin(theta)
    ct = np.cos(theta)
    st_pw = np.sin(theta + w * dt)
    ct_pw = np.cos(theta + w * dt)

    V = np.array([[1./w * (-st + st_pw), v/(w*w) * (st - st_pw) + v/w * ct_pw*dt],
                  [1./w * (ct-ct_pw), -v/(w*w)*(ct - ct_pw) + v/w * (st_pw*dt)],
                  [0., dt]])
    return V

def dynamics(dt, x, u, noise, alphas=[0.01, 0.01, 0.001, 0.001]):
    noise_v = (np.random.randn() * (alphas[0] * u[0] ** 2 + alphas[1] * u[1] ** 2)) if noise else 0
    noise_omega = (np.random.randn() * (alphas[2] * u[0] ** 2 + alphas[3] * u[1] ** 2)) if noise else 0
    v, omega = u[0] + noise_v, u[1] + noise_omega

    px, py, theta = x[0], x[1], x[2]
    px += v / omega * np.sin(theta + omega * dt) - v / omega * np.sin(theta)
    py += v / omega * np.cos(theta) - v / omega * np.cos(theta + omega * dt)
    theta += omega * dt

    dfdx = np.eye(3)
    dfdx[0:2,2] = np.array([v / omega * (-np.cos(x[2]) + np.cos(x[2] + omega * dt)),
                            v / omega * (-np.sin(x[2]) + np.sin(x[2] + omega * dt))])

    return np.array([px, py, theta] + ([0] * (2 * len(LANDMARKS)))), dfdx


def control(t, x):
    v = 1 + 0.5 * np.cos(2 * np.pi * 0.2 * t)
    omega = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t)
    return np.array([v, omega])


class EKFSLAM:
    def __init__(self, Q):
        self.state_dim = 3 + 2 * len(LANDMARKS)
        self.x_hat = np.zeros([self.state_dim])
        self.F = np.zeros([3, self.state_dim])
        self.F[0:3, 0:3] = np.eye(3)
        self.Q = Q
        self.P = scipy.linalg.block_diag(np.eye(3) * 0.0, np.eye(self.state_dim - 3) * 1e10)

    def propagate(self, dt, u, dynamics):
        self.x_hat, dfdx = dynamics(dt, self.x_hat, u, noise=False)

        G = np.eye(self.state_dim) + self.F.T.dot(dfdx).dot(self.F)
        self.P = G.dot(self.P).dot(G.T) + self.F.T.dot(np.eye(3)).dot(self.F)

        return self.x_hat, self.P

    def update(self, dt, z, measure):

        return self.x_hat, self.P

Q = np.array([[0.01], [0.005]]) * np.eye(2)
x = np.array([0, 0, 0])
x_hat = x.copy()
estimator = EKFSLAM(Q=Q)
history = []

for t in time:
    u = control(t, x_hat)
    x, dfdx = dynamics(dt, x, u, noise=False)
    z = measure(dt, x, Q, noise=False)

    x_hat, P = estimator.propagate(dt, u, dynamics)
    x_hat, P = estimator.update(dt, z, measure)

    history.append((x.copy(), x_hat.copy(), np.diag(P)))


history = np.array(history)

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(time, history[:, 0, 0], '-b', label="x")
ax1.plot(time, history[:, 1, 0], '-r', label="xhat")
ax1.plot(time, history[:, 1, 0] + 3.0 * history[:, 2, 0] ** 0.5, '-k', alpha=0.5, label="upper")
ax1.plot(time, history[:, 1, 0] - 3.0 * history[:, 2, 0] ** 0.5, '-k', alpha=0.5, label="lower")
ax1.legend()

ax2.plot(time, history[:, 0, 1], '-b', label="y")
ax2.plot(time, history[:, 1, 1], '-r', label="yhat")
ax2.plot(time, history[:, 1, 1] + 3.0 * history[:, 2, 1] ** 0.5, '-k', alpha=0.5, label="upper")
ax2.plot(time, history[:, 1, 1] - 3.0 * history[:, 2, 1] ** 0.5, '-k', alpha=0.5, label="lower")
ax2.legend()

ax3.plot(time, history[:, 0, 2], '-b', label=r"$\theta$")
ax3.plot(time, history[:, 1, 2], '-r', label=r"$\theta$")
ax3.plot(time, history[:, 1, 2] + 3.0 * history[:, 2, 2] ** 0.5, '-k', alpha=0.5, label="upper")
ax3.plot(time, history[:, 1, 2] - 3.0 * history[:, 2, 2] ** 0.5, '-k', alpha=0.5, label="lower")
ax3.legend()

plt.show()