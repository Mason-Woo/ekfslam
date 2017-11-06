import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy
from matplotlib.patches import Ellipse

dt = 0.1
time = np.arange(0, 10, dt)
LANDMARKS = (np.random.random((5, 2)) - .5) * 30

np.set_printoptions(linewidth=300)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def measure(dt, x, Q=None, noise=False):
    z = np.zeros([2 * LANDMARKS.shape[0]])
    dhdx = np.zeros([2 * len(LANDMARKS), 3 + 2 * len(LANDMARKS)])

    for i, m in enumerate(LANDMARKS):
        dy = m[1] - x[1]
        dx = m[0] - x[0]
        q = (dx) ** 2 + (dy) ** 2
        z[i * 2] = np.sqrt(q)
        z[i * 2 + 1] = np.arctan2(dy, dx) - x[2]
        z[i * 2:i * 2 + 2] = np.random.multivariate_normal(z[i * 2:i * 2 + 2], Q) if noise else z[i * 2:i * 2 + 2]

        if np.abs(z[i * 2 + 1] - x[2]) > 3 * np.pi:
            z[i * 2:i * 2 + 2] = np.inf

        else:
            dhdx[i * 2:i * 2 + 2, 0:3] = 1. / z[i * 2] * np.array([[-z[i * 2] * dx, -z[i * 2] * dy, 0], [dy, -dx, -1]])
            dhdx[i * 2:i * 2 + 2, 3 + 2 * i:3 + 2 * i + 2] = 1. / z[i * 2] * np.array([[z[i * 2] * dx, z[i * 2] * dy], [-dy, dx]])

    return z, dhdx

def dynamics(dt, x, u, noise, alphas=np.array([0.01, 0.001, 0.01, 0.001])):
    noise_v = (np.random.randn() * (alphas[0] * u[0] ** 2 + alphas[1] * u[1] ** 2)) if noise else 0
    noise_omega = (np.random.randn() * (alphas[2] * u[0] ** 2 + alphas[3] * u[1] ** 2)) if noise else 0
    v, omega = u[0] + noise_v, u[1] + noise_omega
    omega = 1e-5 if np.abs(omega) < 1e-5 else omega

    px, py, theta = x[0], x[1], x[2]
    px += v / omega * np.sin(theta + omega * dt) - v / omega * np.sin(theta)
    py += v / omega * np.cos(theta) - v / omega * np.cos(theta + omega * dt)
    theta += omega * dt

    theta += -2. * np.pi if theta >   np.pi else 0
    theta +=  2. * np.pi if theta <= -np.pi else 0

    dfdx = np.eye(3 + 2 * len(LANDMARKS))
    dfdx[0:2,2] = np.array([v / omega * (-np.cos(x[2]) + np.cos(x[2] + omega * dt)),
                            v / omega * (-np.sin(x[2]) + np.sin(x[2] + omega * dt))])

    dfdu = np.zeros([3 + 2 * len(LANDMARKS), 2])
    dfdu[0:3, 0:2] = np.array([[(-np.sin(theta) + np.sin(theta + omega * dt))/omega, v * (np.sin(theta) - np.sin(theta + omega * dt))/(omega**2) + (v * np.cos(theta + omega * dt) * dt)/omega],
                     [(np.cos(theta) - np.cos(theta + omega * dt))/omega, -v * (np.cos(theta) - np.cos(theta + omega * dt)) / (omega ** 2) + (v * np.sin(theta + omega * dt) * dt) / omega],
                     [0, dt]])

    M = np.array([[alphas[0] * v**2 + alphas[1] * omega ** 2, 0],
                  [0,                                         alphas[2] * v**2 + alphas[3] * omega ** 2]])

    return np.array([px, py, theta] + x[3:].tolist()), dfdx, dfdu, M


def control(t, x):
    v = 1 + 0.5 * np.cos(2 * np.pi * 0.2 * t)
    omega = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t)
    return np.array([v, omega])


class EKFSLAM:
    def __init__(self, Q):
        self.state_dim = 3 + 2 * len(LANDMARKS)
        self.x_hat = np.array(([0] * 3) + ([np.inf] * 2 * len(LANDMARKS)))
        self.Q = Q
        self.P = scipy.linalg.block_diag(np.eye(3) * 0., np.eye(self.state_dim - 3) * 1e15)
        self.I = np.eye(3 + 2 * len(LANDMARKS))

    def propagate(self, dt, u, dynamics):
        self.x_hat, G, V, R = dynamics(dt, self.x_hat, u, noise=False)

        self.P = G.dot(self.P).dot(G.T) + V.dot(R).dot(V.T)

        return self.x_hat, self.P

    def update(self, dt, z, measure, Q):
        zhat, H = measure(dt, self.x_hat)
        residual = (z - zhat).reshape(-1, 2)

        residual[residual[:, 1] > np.pi, 1] -= 2. * np.pi
        residual[residual[:, 1] <= -np.pi, 1] += 2. * np.pi

        residual[residual > 0.05] = 0.05
        residual[residual < -0.05] = -0.05

        for l in range(len(LANDMARKS)):
            if not np.isfinite(self.x_hat[3 + 2 * l:3 + 2 * l + 2]).any():
                self.x_hat[3 + 2 * l] = self.x_hat[0] + z[2 * l] * np.cos(z[2 * l + 1] + self.x_hat[2])
                self.x_hat[3 + 2 * l + 1] = self.x_hat[1] + z[2 * l] * np.sin(z[2 * l + 1] + self.x_hat[2])
                continue

            if not np.isfinite(z[2 * l + 1:2 * l + 2]).all():
                continue

            Hl = H[l * 2:l * 2 + 2]
            psi = Hl.dot(self.P).dot(Hl.T) + Q
            K = self.P.dot(Hl.T).dot(scipy.linalg.pinv(psi))

            self.x_hat += K.dot(residual[l])
            self.P = (self.I - K.dot(Hl)).dot(self.P)

        return self.x_hat, self.P

Q = np.array([[0.01], [0.005]]) * np.eye(2)
x = np.array([0, 0, 0] + LANDMARKS.reshape(-1).tolist())
x_hat = x.copy()
estimator = EKFSLAM(Q=Q)
history = []

for t in time:
    u = control(t, x_hat)
    x, dfdx, dfdu, M = dynamics(dt, x, u, noise=True)
    z, dhdx = measure(dt, x, Q, noise=True)

    x_hat, P = estimator.propagate(dt, u, dynamics)
    x_hat, P = estimator.update(dt, z, measure, Q)

    history.append((x.copy(), x_hat.copy(), np.diag(P)))


history = np.array(history)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
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

ax4.scatter(history[-1, 0, 3::2], history[-1, 0, 4::2], c='r', label='truth')

for j in range(len(LANDMARKS)):
    cov = P[3 + j * 2:3 + j * 2 + 2, 3 + j * 2:3 + j * 2 + 2]
    plot_cov_ellipse(cov, history[-1, 1, 3 + j * 2:3 + j * 2 + 2], nstd=2, ax=ax4)

ax4.scatter(history[-1, 1, 3::2],  history[-1, 1, 4::2], c='g', label='estimated')
ax4.scatter(history[:, 0, 0], history[:, 0, 1], c='b', label='robot')
ax4.legend()

ax5.plot(time, (history[:, 0] - history[:,1]))

plt.show()