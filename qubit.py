import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.special
from matplotlib.widgets import Slider


# =============================================================
# 4-Level Transmon — Rotating Frame
# =============================================================

class FourLevelTransmon:

    def __init__(
        self,
        f01=6.62820323,
        f12=6.32820323,
        f23=6.02820323
    ):

        self.N = 4

        # -----------------------------------------------------
        # Transition frequencies
        # -----------------------------------------------------

        self.f01 = f01
        self.f12 = f12
        self.f23 = f23

        # -----------------------------------------------------
        # Anharmonicity / rotating-frame detunings
        # -----------------------------------------------------

        # Drive is assumed to be at f01
        self.delta2 = self.f12 - self.f01

        # |3> is two transitions above |0>
        self.delta3 = (
            self.f12
            + self.f23
            - 2 * self.f01
        )

        # -----------------------------------------------------
        # Rotating-frame static Hamiltonian
        #
        # H/hbar in GHz
        # -----------------------------------------------------

        self.H0 = np.diag([
            0.0,
            0.0,
            self.delta2,
            self.delta3
        ]).astype(complex)

        # -----------------------------------------------------
        # Transmon matrix-element factors
        #
        # 0 <-> 1 : 1
        # 1 <-> 2 : sqrt(2)
        # 2 <-> 3 : sqrt(3)
        # -----------------------------------------------------

        self.matrix_elements = np.array([
            1.0,
            np.sqrt(2),
            np.sqrt(3)
        ])

        # -----------------------------------------------------
        # sigma-x operators
        # -----------------------------------------------------

        self.sigma_x = np.zeros(
            (self.N, self.N),
            dtype=complex
        )

        for j in range(1, self.N):

            g = self.matrix_elements[j - 1]

            self.sigma_x[j, j - 1] = g
            self.sigma_x[j - 1, j] = g

        # -----------------------------------------------------
        # sigma-y operators
        # -----------------------------------------------------

        self.sigma_y = np.zeros(
            (self.N, self.N),
            dtype=complex
        )

        for j in range(1, self.N):

            g = self.matrix_elements[j - 1]

            self.sigma_y[j, j - 1] = -1j * g
            self.sigma_y[j - 1, j] = 1j * g

    # =========================================================
    # Print spectrum
    # =========================================================

    def print_spectrum(self):

        print("4-level transmon")
        print("----------------")
        print()

        print(f"f01 = {self.f01:.8f} GHz")
        print(f"f12 = {self.f12:.8f} GHz")
        print(f"f23 = {self.f23:.8f} GHz")

        print()

        print(
            f"anharmonicity = "
            f"{self.f12 - self.f01:.8f} GHz"
        )

        print()

        print(
            f"Delta2 = "
            f"{self.delta2:.8f} GHz"
        )

        print(
            f"Delta3 = "
            f"{self.delta3:.8f} GHz"
        )

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(
        self,
        t,
        I,
        Q=None,
        amplitude=1.0,
        carrier_frequency=None,
        phase=0.0,
        initial_state=0
    ):

        """
        Simulate the 4-level transmon in the rotating frame.

        Hamiltonian:

        H/hbar =
            Delta2 |2><2|
          + Delta3 |3><3|
          + Omega_x(t)/2 * X
          + Omega_y(t)/2 * Y

        where

        Omega_x(t) = amplitude * I(t)
        Omega_y(t) = amplitude * Q(t)

        Frequencies are in GHz and time is in ns.
        """

        if carrier_frequency is None:
            carrier_frequency = self.f01

        # -----------------------------------------------------
        # Initial state
        # -----------------------------------------------------

        psi0 = np.zeros(
            self.N,
            dtype=complex
        )

        psi0[initial_state] = 1.0

        # -----------------------------------------------------
        # Convert carrier frequency to angular frequency
        # -----------------------------------------------------

        omega_d = 2 * np.pi * carrier_frequency

        # -----------------------------------------------------
        # Q = 0 if not supplied
        # -----------------------------------------------------

        if Q is None:
            Q = np.zeros_like(t)

        # -----------------------------------------------------
        # Schrödinger equation
        # -----------------------------------------------------

        def rhs(time, psi):

            # Interpolate pulse
            I_t = np.interp(
                time,
                t,
                I
            )

            Q_t = np.interp(
                time,
                t,
                Q
            )

            # -------------------------------------------------
            # Envelope Rabi frequencies
            # -------------------------------------------------

            Omega_x = amplitude * I_t
            Omega_y = amplitude * Q_t

            # -------------------------------------------------
            # Rotating-frame Hamiltonian
            # -------------------------------------------------

            H = (
                self.H0
                + (Omega_x / 2.0) * self.sigma_x
                + (Omega_y / 2.0) * self.sigma_y
            )

            # -------------------------------------------------
            # H is in GHz
            # t is in ns
            #
            # dpsi/dt = -i 2pi H psi
            # -------------------------------------------------

            return -1j * 2 * np.pi * H @ psi

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------

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
    # Populations
    # =========================================================

    @staticmethod
    def populations(psi):

        return np.abs(psi)**2


# =============================================================
# Pulse shapes
# =============================================================


def gaussian(t, T, sigma):

    return np.exp(
        -(t - T / 2)**2
        / (2 * sigma**2)
    )


def square(t, T, width=None):

    if width is None:
        width = T

    center = T / 2

    return np.where(
        np.abs(t - center) <= width / 2,
        1.0,
        0.0
    )


def cosine_squared(t, T):

    x = t / T

    return np.where(
        (x >= 0) & (x <= 1),
        np.sin(np.pi * x)**2,
        0.0
    )


def blackman(t, T):

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


def gaussian_flat_top(
    t,
    T,
    sigma,
    flat_fraction=0.5
):

    flat_width = flat_fraction * T

    t_start = (T - flat_width) / 2
    t_end = (T + flat_width) / 2

    rise = 0.5 * (
        1
        + scipy.special.erf(
            (t - t_start)
            / (np.sqrt(2) * sigma)
        )
    )

    fall = 0.5 * (
        1
        + scipy.special.erf(
            (t_end - t)
            / (np.sqrt(2) * sigma)
        )
    )

    return rise * fall


def drag(
    t,
    T,
    sigma,
    beta
):

    # ---------------------------------------------------------
    # I quadrature
    # ---------------------------------------------------------

    I = gaussian(
        t,
        T,
        sigma
    )

    # ---------------------------------------------------------
    # Derivative
    # ---------------------------------------------------------

    dI_dt = np.gradient(
        I,
        t
    )

    # ---------------------------------------------------------
    # DRAG quadrature
    # ---------------------------------------------------------

    Q = -beta * dI_dt

    return I, Q


# =============================================================
# Pulse selector
# =============================================================


def make_pulse(
    name,
    t,
    T,
    sigma,
    beta=0.0
):

    if name == "gaussian":

        I = gaussian(
            t,
            T,
            sigma
        )

        Q = np.zeros_like(t)

    elif name == "square":

        I = square(
            t,
            T
        )

        Q = np.zeros_like(t)

    elif name == "cosine_squared":

        I = cosine_squared(
            t,
            T
        )

        Q = np.zeros_like(t)

    elif name == "blackman":

        I = blackman(
            t,
            T
        )

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
# Create transmon
# =============================================================

q = FourLevelTransmon(
    f01=6.62820323,
    f12=6.32820323,
    f23=6.02820323
)

q.print_spectrum()


# =============================================================
# Pulse parameters
# =============================================================

T = 20.0          # ns
sigma = T / 6     # ns
T_max = 100.0      # Maximum duration allowed by slider

t = np.linspace(
    0,
    T_max,
    50000
)


# =============================================================
# Initial pulse parameters
# =============================================================

pulse_name = "drag"

amplitude = 0.025   # GHz
beta = 0.5          # ns


# =============================================================
# Generate pulse
# =============================================================

I, Q = make_pulse(
    pulse_name,
    t=t,
    T=T,
    sigma=sigma,
    beta=beta
)


# =============================================================
# Initial simulation
# =============================================================

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


axes[0].set_ylabel(
    "Population"
)

axes[0].set_title(
    "4-level transmon — DRAG"
)

axes[0].set_ylim(
    0,
    1.05
)

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

axes[1].set_ylabel(
    "Rabi amplitude (GHz)"
)

axes[1].grid()
axes[1].legend()


iq_max = max(
    np.max(np.abs(amplitude * I)),
    np.max(np.abs(amplitude * Q)),
    1e-12
)

axes[1].set_ylim(
    -1.15 * iq_max,
    1.15 * iq_max
)


# =============================================================
# Microwave waveform
#
# This is ONLY for visualization.
#
# The simulation itself is performed in the rotating frame.
# =============================================================

carrier = (
    2
    * np.pi
    * q.f01
    * t
)

waveform = amplitude * (
    I * np.cos(carrier)
    + Q * np.sin(carrier)
)

line_waveform, = axes[2].plot(
    t,
    waveform
)

axes[2].set_xlabel(
    "Time (ns)"
)

axes[2].set_ylabel(
    "Drive (GHz)"
)

axes[2].grid()


waveform_max = max(
    np.max(np.abs(waveform)),
    1e-12
)

axes[2].set_ylim(
    -1.15 * waveform_max,
    1.15 * waveform_max
)


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
# Slider: DRAG beta
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
# Slider: pulse duration
# =============================================================

ax_T = plt.axes([
    0.15,
    0.13,
    0.70,
    0.03
])

slider_T = Slider(
    ax_T,
    "Pulse time (ns)",
    5.0,
    T_max,
    valinit=T,
    valstep=0.1
)


# =============================================================
# Live numerical readout
# =============================================================

info_text = fig.text(
    0.02,
    0.955,
    "",
    fontsize=10
)


# =============================================================
# Update function
# =============================================================

def update(val):

    amplitude = slider_amp.val
    beta = slider_beta.val
    T = slider_T.val

    # ---------------------------------------------------------
    # Pulse width scales with pulse duration
    # ---------------------------------------------------------

    sigma = T / 6

    # ---------------------------------------------------------
    # Generate new pulse
    # ---------------------------------------------------------

    I, Q = make_pulse(
        "drag",
        t=t,
        T=T,
        sigma=sigma,
        beta=beta
    )

    # ---------------------------------------------------------
    # Simulate
    # ---------------------------------------------------------

    psi = q.simulate(
        t=t,
        I=I,
        Q=Q,
        amplitude=amplitude,
        carrier_frequency=q.f01
    )

    P = q.populations(psi)

    # ---------------------------------------------------------
    # Update populations
    # ---------------------------------------------------------

    for i in range(4):

        lines[i].set_ydata(
            P[:, i]
        )

    axes[0].set_ylim(
        0,
        1.05
    )

    # ---------------------------------------------------------
    # Update I/Q
    # ---------------------------------------------------------

    I_plot = amplitude * I
    Q_plot = amplitude * Q

    line_I.set_ydata(I_plot)
    line_Q.set_ydata(Q_plot)

    iq_max = max(
        np.max(np.abs(I_plot)),
        np.max(np.abs(Q_plot)),
        1e-12
    )

    axes[1].set_ylim(
        -1.15 * iq_max,
        1.15 * iq_max
    )

    # ---------------------------------------------------------
    # Update microwave waveform
    # ---------------------------------------------------------

    waveform = amplitude * (
        I * np.cos(carrier)
        + Q * np.sin(carrier)
    )

    line_waveform.set_ydata(
        waveform
    )

    waveform_max = max(
        np.max(np.abs(waveform)),
        1e-12
    )

    axes[2].set_ylim(
        -1.15 * waveform_max,
        1.15 * waveform_max
    )

    # ---------------------------------------------------------
    # Update title
    # ---------------------------------------------------------

    axes[0].set_title(
        f"4-level transmon — DRAG "
        f"(T = {T:.1f} ns)"
    )

    # ---------------------------------------------------------
    # Live numerical readout
    # ---------------------------------------------------------

    final = P[-1]

    max_P2 = np.max(P[:, 2])
    max_P3 = np.max(P[:, 3])

    leakage = final[2] + final[3]

    info_text.set_text(
        f"A = {amplitude:.3f} GHz    "
        f"β = {beta:.2f} ns    "
        f"T = {T:.1f} ns    "
        f"P0 = {final[0]:.4f}    "
        f"P1 = {final[1]:.4f}    "
        f"P2 = {final[2]:.4e}    "
        f"P3 = {final[3]:.4e}    "
        f"Leakage = {leakage:.4e}"
    )

    fig.canvas.draw_idle()


# =============================================================
# Connect sliders
# =============================================================

slider_amp.on_changed(
    update
)

slider_beta.on_changed(
    update
)

slider_T.on_changed(
    update
)


# =============================================================
# Show
# =============================================================

plt.show()
