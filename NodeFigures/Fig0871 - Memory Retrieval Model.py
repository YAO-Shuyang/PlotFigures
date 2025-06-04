import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft

# Parameters
N = 512
alpha = 0.03
mbar = 1
J0 = 1
J0 = J0 / N * 512
a = 0.15
rho = N / (2 * np.pi)
kc = rho * J0**2 / (8 * np.sqrt(2) * np.pi * a)
k = 0.1 * kc

tau = 3
tau_v = 144
dt = tau / 10
m = mbar * tau / tau_v

# Initialization
J = np.zeros(N)
U = np.zeros(N)
V = np.zeros(N)
r = np.zeros(N)

# Positions
pos = np.linspace(-np.pi, np.pi, N, endpoint=False)

# Weight matrix construction
for i in range(N):
    dx = min(abs(pos[i] - pos[0]), np.pi - abs(pos[i]))
    J[i] = J0 / (np.sqrt(2 * np.pi) * a) * np.exp(-dx**2 / (2 * a**2))

Jfft = fft(J)

# Simulation parameters
T = 1000
v = 2e-3
times = np.arange(0, T, dt)

fig, ax = plt.subplots(figsize=(8, 6))
line_r, = ax.plot([], [], 'b', linewidth=3, label='Firing rate (r)')
line_Iext, = ax.plot([], [], 'r', linewidth=3, label='External input (Iext)')
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([0, 0.04])
ax.set_xlabel('Position (rad)')
ax.set_ylabel('Activity')
ax.grid(False)
ax.legend()

# Animation function
def update(frame):
    global U, V, r
    t = times[frame]

    if t <= 50:
        Iext = alpha * np.exp(-pos**2 / (2 * a**2))
    else:
        x_stim = v * (t - 50)
        dx = np.minimum(np.abs(x_stim - pos), 2 * np.pi - np.abs(x_stim - pos))
        Iext = alpha * np.exp(-dx**2 / (2 * a**2))

    Irec = np.real(ifft(Jfft * fft(r)))

    dU = dt * (-U - V + Iext + Irec) / tau
    U += dU
    dV = dt * (-V + m * U) / tau_v
    V += dV

    U = np.maximum(U, 0)
    r = U**2 / (1 + k * np.sum(U**2))

    line_r.set_data(pos, r)
    line_Iext.set_data(pos, Iext)
    ax.set_title(f'Time = {t*1000:.3f}')

    return line_r, line_Iext

anim = FuncAnimation(fig, update, frames=len(times), interval=10, blit=True)

plt.show()
