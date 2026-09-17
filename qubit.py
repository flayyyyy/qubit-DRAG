import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.special
from matplotlib.widgets import Slider


class FourLevelTransmon:

    def __init__(self, Ej=20.0, Ec=0.3):

        self.Ej = Ej
        self.Ec = Ec
        self.N = 4

        # ---------------------------------------------
        # Transmon frequencies
        # ---------------------------------------------

        self.f_p = np.sqrt(8 * Ej * Ec)

        # First-order transmon energies
        m = np.arange(self.N)

        self.energies = (
            -Ej
            + self.f_p * (m + 0.5)
            - Ec / 12.0 * (
                6 * m**2 + 6 * m + 3
            )
        )

        # ---------------------------------------------
        # Static Hamiltonian
        # ---------------------------------------------

        self.H0 = np.diag(self.energies)

        # ---------------------------------------------
        # Harmonic oscillator operators
        # ---------------------------------------------

        a = np.zeros((self.N, self.N), dtype=complex)

        for n in range(1, self.N):
            a[n - 1, n] = np.sqrt(n)

        adag = a.conj().T

        # ---------------------------------------------
        # Zero-point fluctuations
        # ---------------------------------------------

        self.phi_zpf = (2 * Ec / Ej) ** 0.25
        self.n_zpf = (Ej / (32 * Ec)) ** 0.25

        # ---------------------------------------------
        # Charge operator
        # ---------------------------------------------

        self.n = (
            1j
            * self.n_zpf
            * (adag - a)
        )

        # Remove global ground-state energy.
        # This is convenient for numerical evolution.
        self.H0 = self.H0 - self.energies[0] * np.eye(self.N)

        self.energies = self.energies - self.energies[0]

        # Transition frequencies
        self.f01 = self.energies[1] - self.energies[0]
        self.f12 = self.energies[2] - self.energies[1]
        self.f23 = self.energies[3] - self.energies[2]

    # =========================================================
    # Arbitrary pulse simulation
    # =========================================================

    def simulate(
        self,
        t,
        I,
        Q=None,
        amplitude=1,
        carrier_frequency=None,
        phase=0.0,
        initial_state=0
    ):

        if carrier_frequency is None:
            carrier_frequency = self.f01

        # ---------------------------------------------------------
        # Initial state
        # ---------------------------------------------------------

        psi0 = np.zeros(self.N, dtype=complex)
        psi0[initial_state] = 1.0

        # ---------------------------------------------------------
        # Time-dependent Schrodinger equation
        # ---------------------------------------------------------

        def rhs(time, psi):

            # Interpolate pulse values because solve_ivp can
            # evaluate between t_eval points.
            I_t = np.interp(time, t, I)

            if Q is None:
                Q_t = 0.0
            else:
                Q_t = np.interp(time, t, Q)

            # Carrier
            theta = (
                2 * np.pi
                * carrier_frequency
                * time
                + phase
            )

            drive = amplitude * (
                I_t * np.cos(theta)
                + Q_t * np.sin(theta)
            )

            H = self.H0 + drive * self.n

            # H is in GHz, t is in ns
            return -1j * 2 * np.pi * H @ psi

        # ---------------------------------------------------------
        # Solve
        # ---------------------------------------------------------

        result = solve_ivp(
            rhs,
            (t[0], t[-1]),
            psi0,
            t_eval=t,
            rtol=1e-9,
            atol=1e-11
        )

        return result.y.T

    # =========================================================
    # Population
    # =========================================================

    @staticmethod
    def populations(psi):

        return np.abs(psi)**2

    # =========================================================
    # Print spectrum
    # =========================================================

    def print_spectrum(self):

        print("4-level transmon")
        print("----------------")
        print(f"Ej = {self.Ej}")
        print(f"Ec = {self.Ec}")
        print()

        print(f"f01 = {self.f01:.8f}")
        print(f"f12 = {self.f12:.8f}")
        print(f"f23 = {self.f23:.8f}")
        print()

        print(
            f"anharmonicity = "
            f"{self.f12 - self.f01:.8f}"
        )

# =============================================================
# Create transmon
# =============================================================

q = FourLevelTransmon(
    Ej=20.0,
    Ec=0.3
)

q.print_spectrum()


# =============================================================
# Pulse parameters
# =============================================================

T=20       # ns
sigma=T/6  # ns

t = np.linspace(
    0,
    T,
    10000
)

amplitude=0.05

# =============================================================
# Pulse shapes
# =============================================================

def gaussian(t, T, sigma):
    """
    Gaussian pulse centered at T/2.
    """
    return np.exp(
        -(t - T/2)**2 / (2 * sigma**2)
    )


def square(t, T, width=None):
    """
    Centered square pulse.

    If width is None, the pulse occupies the full duration T.
    """
    if width is None:
        width = T

    center = T / 2

    return np.where(
        np.abs(t - center) <= width / 2,
        1.0,
        0.0
    )


def cosine_squared(t, T):
    """
    sin^2 pulse with zero amplitude at both ends.
    """
    x = t / T

    return np.where(
        (x >= 0) & (x <= 1),
        np.sin(np.pi * x)**2,
        0.0
    )


def blackman(t, T):
    """
    Blackman window.
    """
    x = t / T

    return np.where(
        (x >= 0) & (x <= 1),
        (
            0.42
            - 0.5 * np.cos(2 * np.pi * x)
            + 0.08 * np.cos(4 * np.pi * x)
        ),
        0.0
    )


def gaussian_flat_top(t, T, sigma, flat_fraction=0.5):
    """
    Gaussian flat-top pulse.

    The edges are smooth Gaussian-like ramps and the
    middle is flat.
    """

    flat_width = flat_fraction * T

    t_start = (T - flat_width) / 2
    t_end = (T + flat_width) / 2

    # Smooth Gaussian edges
    rise = 0.5 * (
        1 + scipy.special.erf(
            (t - t_start) / (np.sqrt(2) * sigma)
        )
    )

    fall = 0.5 * (
        1 + scipy.special.erf(
            (t_end - t) / (np.sqrt(2) * sigma)
        )
    )

    return rise * fall


def drag(t, T, sigma, beta):
    """
    Gaussian DRAG pulse.

    Returns:
        I(t), Q(t)
    """

    I = gaussian(t, T, sigma)

    dI_dt = np.gradient(I, t)

    Q = -beta * dI_dt

    return I, Q

# =============================================================
# Pulse Selector
# =============================================================

def make_pulse(
    name,
    t,
    T,
    sigma,
    beta=0.0
):

    if name == "gaussian":

        I = gaussian(t, T, sigma)
        Q = np.zeros_like(t)

    elif name == "square":

        I = square(t, T)
        Q = np.zeros_like(t)

    elif name == "cosine_squared":

        I = cosine_squared(t, T)
        Q = np.zeros_like(t)

    elif name == "blackman":

        I = blackman(t, T)
        Q = np.zeros_like(t)

    elif name == "gaussian_flat_top":

        I = gaussian_flat_top(
            t,
            T,
            sigma,
            flat_fraction=0.5
        )

        Q = np.zeros_like(t)

    elif name == "drag":

        I, Q = drag(
            t,
            T,
            sigma,
            beta
        )

    else:

        raise ValueError(
            f"Unknown pulse: {name}"
        )

    return I, Q

# =============================================================
# Interactive DRAG simulation
# =============================================================

pulse_name = "drag"

# Initial values
amplitude = 0.025
beta = 0.5

I, Q = make_pulse(
    pulse_name,
    t=t,
    T=T,
    sigma=sigma,
    beta=beta
)

psi = q.simulate(
    t=t,
    I=I,
    Q=Q,
    amplitude=amplitude,
    carrier_frequency=q.f01
)

P = q.populations(psi)


# =============================================================
# Figure
# =============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True
)

plt.subplots_adjust(
    bottom=0.20,
    hspace=0.35
)


# =============================================================
# Population plot
# =============================================================

lines = []

for i in range(4):

    line, = axes[0].plot(
        t,
        P[:, i],
        label=f"|{i}>"
    )

    lines.append(line)

axes[0].set_ylabel("Population")
axes[0].set_title("4-level transmon — DRAG")
axes[0].set_ylim(0, 1.05)
axes[0].grid()
axes[0].legend()


# =============================================================
# I/Q plot
# =============================================================

line_I, = axes[1].plot(
    t,
    amplitude * I,
    label="I(t)"
)

line_Q, = axes[1].plot(
    t,
    amplitude * Q,
    label="Q(t)"
)

axes[1].set_ylabel("Amplitude (GHz)")
axes[1].grid()
axes[1].legend()


# =============================================================
# Microwave waveform
# =============================================================

carrier = 2 * np.pi * q.f01 * t

waveform = amplitude * (
    I * np.cos(carrier)
    + Q * np.sin(carrier)
)

line_waveform, = axes[2].plot(
    t,
    waveform
)

axes[2].set_xlabel("Time (ns)")
axes[2].set_ylabel("Drive (GHz)")
axes[2].grid()


# =============================================================
# Slider: amplitude
# =============================================================

ax_amp = plt.axes([
    0.15,
    0.08,
    0.70,
    0.03
])

slider_amp = Slider(
    ax_amp,
    "Amplitude (GHz)",
    0.001,
    0.10,
    valinit=amplitude,
    valstep=0.001
)


# =============================================================
# Slider: beta
# =============================================================

ax_beta = plt.axes([
    0.15,
    0.03,
    0.70,
    0.03
])

slider_beta = Slider(
    ax_beta,
    "DRAG β (ns)",
    -5.0,
    5.0,
    valinit=beta,
    valstep=0.05
)


# =============================================================
# Update function
# =============================================================

def update(val):

    amplitude = slider_amp.val
    beta = slider_beta.val

    # =========================================================
    # Generate new DRAG pulse
    # =========================================================

    I, Q = make_pulse(
        "drag",
        t=t,
        T=T,
        sigma=sigma,
        beta=beta
    )

    # =========================================================
    # Simulate
    # =========================================================

    psi = q.simulate(
        t=t,
        I=I,
        Q=Q,
        amplitude=amplitude,
        carrier_frequency=q.f01
    )

    P = q.populations(psi)

    # =========================================================
    # Update populations
    # =========================================================

    for i in range(4):
        lines[i].set_ydata(P[:, i])

    # Keep population axis fixed
    axes[0].set_ylim(0, 1.05)

    # =========================================================
    # Update I/Q
    # =========================================================

    I_plot = amplitude * I
    Q_plot = amplitude * Q

    line_I.set_ydata(I_plot)
    line_Q.set_ydata(Q_plot)

    # Dynamically scale I/Q axis
    iq_max = max(
        np.max(np.abs(I_plot)),
        np.max(np.abs(Q_plot)),
        1e-12
    )

    axes[1].set_ylim(
        -1.15 * iq_max,
        1.15 * iq_max
    )

    # =========================================================
    # Update microwave waveform
    # =========================================================

    waveform = amplitude * (
        I * np.cos(carrier)
        + Q * np.sin(carrier)
    )

    line_waveform.set_ydata(waveform)

    # Dynamically scale waveform axis
    waveform_max = max(
        np.max(np.abs(waveform)),
        1e-12
    )

    axes[2].set_ylim(
        -1.15 * waveform_max,
        1.15 * waveform_max
    )

    # =========================================================
    # Redraw
    # =========================================================

    fig.canvas.draw_idle()


# =============================================================
# Connect sliders
# =============================================================

slider_amp.on_changed(update)
slider_beta.on_changed(update)

plt.show()







# # =============================================================
# # Generate DRAG pulse
# # =============================================================

# I, Q = make_pulse(
#     "drag",
#     t=t,
#     T=T,
#     sigma=sigma,
#     beta=0.5
# )


# # =============================================================
# # Sweep drive amplitude
# # =============================================================

# amplitudes = np.linspace(0.01, 0.5, 100)

# final_P0 = []
# final_P1 = []
# final_P2 = []
# final_P3 = []

# max_P2 = []
# max_P3 = []


# for A in amplitudes:

#     psi = q.simulate(
#         t=t,
#         I=I,
#         Q=Q,
#         amplitude=A,
#         carrier_frequency=q.f01
#     )

#     P = q.populations(psi)

#     final_P0.append(P[-1, 0])
#     final_P1.append(P[-1, 1])
#     final_P2.append(P[-1, 2])
#     final_P3.append(P[-1, 3])

#     max_P2.append(np.max(P[:, 2]))
#     max_P3.append(np.max(P[:, 3]))


# # =============================================================
# # Convert lists to numpy arrays
# # =============================================================

# final_P0 = np.array(final_P0)
# final_P1 = np.array(final_P1)
# final_P2 = np.array(final_P2)
# final_P3 = np.array(final_P3)

# max_P2 = np.array(max_P2)
# max_P3 = np.array(max_P3)


# # =============================================================
# # Plot final populations
# # =============================================================

# plt.figure(figsize=(10, 6))

# plt.plot(
#     amplitudes,
#     final_P0,
#     label="P0"
# )

# plt.plot(
#     amplitudes,
#     final_P1,
#     label="P1"
# )

# plt.plot(
#     amplitudes,
#     final_P2,
#     label="P2"
# )

# plt.plot(
#     amplitudes,
#     final_P3,
#     label="P3"
# )

# plt.xlabel("Drive amplitude (GHz)")
# plt.ylabel("Final population")

# plt.title(
#     "Final population vs drive amplitude — DRAG"
# )

# plt.grid()
# plt.legend()

# plt.tight_layout()
# plt.show()
