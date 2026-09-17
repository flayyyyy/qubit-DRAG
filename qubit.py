import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from qutip import Bloch

from matplotlib import cm

# Fixed colour normalization
nrm = mpl.colors.LogNorm(vmin=1e-3, vmax=1)
cmap = cm.inferno


# =============================================================
# 3-LEVEL TRANSMON
# =============================================================

class ThreeLevelTransmon:

    def __init__(
        self,
        f01=5.11722,             # GHz
        anharmonicity=-0.31528   # GHz
    ):
        self.f01 = f01
        self.anharmonicity = anharmonicity

        # In the rotating frame at f01:
        #
        # E0 = 0
        # E1 = 0
        # E2 = Delta
        #
        # Convert GHz -> rad/ns
        self.Delta = 2 * np.pi * anharmonicity

        # -----------------------------------------------------
        # Static Hamiltonian
        # -----------------------------------------------------

        self.H0 = np.diag([
            0.0,
            0.0,
            self.Delta
        ]).astype(complex)

        # -----------------------------------------------------
        # Transition operators
        # -----------------------------------------------------

        # sigma_x01
        self.sx01 = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=complex)

        # sigma_x12
        self.sx12 = np.array([
            [0, 0, 0],
            [0, 0, np.sqrt(2)],
            [0, np.sqrt(2), 0]
        ], dtype=complex)

        # sigma_y01
        self.sy01 = np.array([
            [0, -1j, 0],
            [1j, 0, 0],
            [0, 0, 0]
        ], dtype=complex)

        # sigma_y12
        self.sy12 = np.array([
            [0, 0, 0],
            [0, 0, -1j*np.sqrt(2)],
            [0, 1j*np.sqrt(2), 0]
        ], dtype=complex)

        # -----------------------------------------------------
        # Operators appearing in the paper's Hamiltonian
        # -----------------------------------------------------

        self.Sx = self.sx01 + self.sx12
        self.Sy = self.sy01 + self.sy12

    # =========================================================
    # Hamiltonian
    # =========================================================

    def hamiltonian(self, Omega_x, Omega_y):
        """
        Rotating-frame Hamiltonian:

        H/hbar =
            Delta |2><2|
            + Omega_x/2 * (sigma_x01 + sigma_x12)
            + Omega_y/2 * (sigma_y01 + sigma_y12)

        Omega_x and Omega_y are supplied in GHz.
        Internally they are converted to rad/ns.
        """

        # GHz -> rad/ns
        Omega_x_rad = 2 * np.pi * Omega_x
        Omega_y_rad = 2 * np.pi * Omega_y

        H = (
            self.H0
            + 0.5 * Omega_x_rad * self.Sx
            + 0.5 * Omega_y_rad * self.Sy
        )

        return H

    # =========================================================
    # Time evolution
    # =========================================================

    def simulate(
        self,
        t,
        Omega_x,
        Omega_y,
        initial_state=0
    ):
        """
        Solve the time-dependent Schrodinger equation.

        d|psi>/dt = -i H(t)|psi>
        """

        psi0 = np.zeros(3, dtype=complex)
        psi0[initial_state] = 1.0

        def rhs(time, psi):

            # Interpolate IQ pulses at solver time
            Ox = np.interp(time, t, Omega_x)
            Oy = np.interp(time, t, Omega_y)

            H = self.hamiltonian(Ox, Oy)

            return -1j * H @ psi

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

    # =========================================================
    # Spectrum
    # =========================================================

    def print_spectrum(self):

        print("3-level transmon")
        print("----------------")
        print(f"f01 = {self.f01:.5f} GHz")
        print(
            f"Delta/2pi = "
            f"{self.anharmonicity:.5f} GHz"
        )
        print(
            f"f12 = "
            f"{self.f01 + self.anharmonicity:.5f} GHz"
        )
        print()


# =============================================================
# PULSE DEFINITIONS
# =============================================================

def gaussian(t, T, sigma):
    """
    Gaussian envelope centered at T/2.
    """

    return np.exp(
        -(t - T / 2)**2
        / (2 * sigma**2)
    )


def gaussian_derivative(t, T, sigma):
    """
    dG/dt
    """

    G = gaussian(t, T, sigma)

    return -(
        (t - T / 2)
        / sigma**2
    ) * G


def drag_pulse(
    t,
    T,
    sigma,
    amplitude,
    beta
):
    """
    DRAG pulse:

        Omega_x = A G(t)

        Omega_y = -A beta dG/dt

    beta has units of time (ns).
    """

    G = gaussian(
        t,
        T,
        sigma
    )

    dG = gaussian_derivative(
        t,
        T,
        sigma
    )

    Omega_x = amplitude * G
    Omega_y = -amplitude * beta / q.anharmonicity * dG

    return Omega_x, Omega_y


# =============================================================
# CREATE TRANSMON
# =============================================================

q = ThreeLevelTransmon(
    f01=5.11722,
    anharmonicity=-0.31528
)

q.print_spectrum()


# =============================================================
# INITIAL PARAMETERS
# =============================================================

T_max = 20.0       # ns

T = 4.6           # ns
sigma_fraction = 6

amplitude = 0.2715   # GHz
beta = 0.086         # ns


# =============================================================
# TIME AXIS
# =============================================================

N = 5000

t = np.linspace(
    0,
    T_max,
    N
)


# =============================================================
# CREATE INITIAL PULSE
# =============================================================

sigma = T / sigma_fraction

Omega_x = np.zeros_like(t)
Omega_y = np.zeros_like(t)

# Only generate pulse inside [0,T]
pulse_mask = t <= T

Omega_x[pulse_mask], Omega_y[pulse_mask] = drag_pulse(
    t[pulse_mask],
    T,
    sigma,
    amplitude,
    beta
)


# =============================================================
# INITIAL SIMULATION
# =============================================================

psi = q.simulate(
    t=t,
    Omega_x=Omega_x,
    Omega_y=Omega_y
)

P = q.populations(psi)


# =============================================================
# FIGURE
# =============================================================

fig = plt.figure(figsize=(12, 7))

# Left side: three time-domain plots
gs = fig.add_gridspec(
    3,
    2,
    width_ratios=[1.5, 1.0],
    hspace=0.35,
    wspace=0.25
)

ax_pop = fig.add_subplot(gs[0, 0])
ax_iq = fig.add_subplot(gs[1, 0])
ax_drive = fig.add_subplot(gs[2, 0])

# Right side: Bloch sphere
ax_bloch = fig.add_axes(
    [0.67, 0.38, 0.30, 0.52],
    projection="3d"
)
cax = fig.add_axes([0.58, 0.43, 0.01, 0.43])

sm = mpl.cm.ScalarMappable(norm=nrm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(r"Leakage $P_2$")

# =============================================================
# BLOCH SPHERE
# =============================================================

b = Bloch(fig=fig, axes=ax_bloch)

b.sphere_alpha = 0.05
b.sphere_color = "#0091ff"

ax_bloch.view_init(elev=0, azim=90)

# Initial Bloch trajectory
c0 = psi[:, 0]
c1 = psi[:, 1]

bx = 2 * np.real(np.conj(c0) * c1)
by = 2 * np.imag(np.conj(c0) * c1)
bz = np.abs(c0)**2 - np.abs(c1)**2

b.add_points([bx, by, bz])
b.point_color
b.point_size = [0.2]
b.vector_color = ["black"]
b.vector_width = 2
b.add_vectors([
    bx[-1],
    by[-1],
    bz[-1]
])
colors = [to_hex(cmap(nrm(x))) for x in P[:, 2]]
b.point_color = colors

b.render()


# =============================================================
# AXIS 1: POPULATIONS
# =============================================================

line_P0, = ax_pop.plot(
    t,
    P[:, 0],
    label=r"$|0\rangle$"
)

line_P1, = ax_pop.plot(
    t,
    P[:, 1],
    label=r"$|1\rangle$"
)

line_P2, = ax_pop.plot(
    t,
    P[:, 2],
    label=r"$|2\rangle$"
)

ax_pop.set_ylabel("Population")
ax_pop.set_ylim(0, 1.05)
ax_pop.grid()
ax_pop.legend(loc="upper right")


# =============================================================
# AXIS 2: IQ COMPONENTS
# =============================================================

line_I, = ax_iq.plot(
    t,
    Omega_x,
    label=r"$\Omega_x(t)$"
)

line_Q, = ax_iq.plot(
    t,
    Omega_y,
    label=r"$\Omega_y(t)$"
)

ax_iq.set_ylabel("IQ amplitude (GHz)")
ax_iq.grid()
ax_iq.legend(loc="upper right")


# =============================================================
# AXIS 3: ACTUAL MICROWAVE DRIVE
# =============================================================

# The physical carrier is reconstructed only for plotting.
#
# drive(t) = Omega_x cos(2*pi*f01*t)
#            - Omega_y sin(2*pi*f01*t)

ax_drive.set_xlabel("Time (ns)")
ax_drive.set_ylabel("Drive (GHz)")
ax_drive.grid()

omega01 = 2 * np.pi * q.f01

drive = (
    Omega_x * np.cos(omega01 * t)
    - Omega_y * np.sin(omega01 * t)
)

line_drive, = ax_drive.plot(
    t,
    drive,
    linewidth=0.8
)


# =============================================================
# SLIDER AXES
# =============================================================

ax_T = plt.axes([
    0.65,     # x
    0.28,     # y
    0.30,     # width
    0.025     # height
])

ax_A = plt.axes([
    0.65,
    0.225,
    0.30,
    0.025
])

ax_beta = plt.axes([
    0.65,
    0.17,
    0.30,
    0.025
])


# =============================================================
# SLIDERS
# =============================================================

slider_T = Slider(
    ax_T,
    "Pulse time (ns)",
    1.0,
    T_max,
    valinit=T,
    valstep=0.1
)

slider_A = Slider(
    ax_A,
    "Amplitude (GHz)",
    0.001,
    1,
    valinit=amplitude,
    valstep=0.0005
)

slider_beta = Slider(
    ax_beta,
    "- DRAG beta (ns)",
    0.0,
    0.5,
    valinit=beta,
    valstep=0.0005
)


# =============================================================
# UPDATE FUNCTION
# =============================================================

def update(val):

    T_new = slider_T.val
    A_new = slider_A.val
    beta_new = slider_beta.val

    sigma_new = T_new / sigma_fraction

    # ---------------------------------------------------------
    # Recalculate pulse
    # ---------------------------------------------------------

    Ox_new = np.zeros_like(t)
    Oy_new = np.zeros_like(t)

    mask = t <= T_new

    Ox_new[mask], Oy_new[mask] = drag_pulse(
        t[mask],
        T_new,
        sigma_new,
        A_new,
        beta_new
    )

    # ---------------------------------------------------------
    # Recalculate evolution
    # ---------------------------------------------------------

    psi_new = q.simulate(
        t=t,
        Omega_x=Ox_new,
        Omega_y=Oy_new
    )

    P_new = q.populations(psi_new)

    # ---------------------------------------------------------
    # Update Bloch sphere
    # ---------------------------------------------------------

    c0 = psi_new[:, 0]
    c1 = psi_new[:, 1]

    bx = 2 * np.real(np.conj(c0) * c1)
    by = 2 * np.imag(np.conj(c0) * c1)
    bz = np.abs(c0)**2 - np.abs(c1)**2

    b.clear()

    b.add_points([bx, by, bz])
    b.point_size = [0.2]
    b.add_vectors([
        bx[-1],
        by[-1],
        bz[-1]
    ])
    b.vector_color = ["black"]
    b.vector_width = 2

    colors = [to_hex(cmap(nrm(x))) for x in P_new[:, 2]]
    b.point_color = colors

    b.render()

    # ---------------------------------------------------------
    # Update populations
    # ---------------------------------------------------------

    line_P0.set_ydata(P_new[:, 0])
    line_P1.set_ydata(P_new[:, 1])
    line_P2.set_ydata(P_new[:, 2])

    # ---------------------------------------------------------
    # Update IQ
    # ---------------------------------------------------------

    line_I.set_ydata(Ox_new)
    line_Q.set_ydata(Oy_new)

    # ---------------------------------------------------------
    # Update physical carrier
    # ---------------------------------------------------------

    drive_new = (
        Ox_new * np.cos(omega01 * t)
        - Oy_new * np.sin(omega01 * t)
    )

    line_drive.set_ydata(drive_new)

    # ---------------------------------------------------------
    # Dynamically update y limits
    # ---------------------------------------------------------

    iq_max = max(
        np.max(np.abs(Ox_new)),
        np.max(np.abs(Oy_new)),
        1e-6
    )

    ax_iq.set_ylim(
        -1.2 * iq_max,
        1.2 * iq_max
    )

    drive_max = max(
        np.max(np.abs(drive_new)),
        1e-6
    )

    ax_drive.set_ylim(
        -1.2 * drive_max,
        1.2 * drive_max
    )

    # ---------------------------------------------------------
    # Update title
    # ---------------------------------------------------------

    final = P_new[-1]

    leakage = final[2]

    ax_pop.set_title(
        "3-level transmon — DRAG   "
        f"(T = {T_new:.1f} ns, "
        f"A = {A_new:.4f} GHz, "
        f"β = {beta_new:.2f} ns)\n"
        f"P0 = {final[0]:.5f}   "
        f"P1 = {final[1]:.5f}   "
        f"P2 = {final[2]:.3e}   "
        f"Leakage = {leakage:.3e}"
    )

    fig.canvas.draw_idle()


# =============================================================
# CONNECT SLIDERS
# =============================================================

slider_T.on_changed(update)
slider_A.on_changed(update)
slider_beta.on_changed(update)


# =============================================================
# INITIAL Y LIMITS
# =============================================================

iq_max = max(
    np.max(np.abs(Omega_x)),
    np.max(np.abs(Omega_y))
)

ax_iq.set_ylim(
    -1.2 * iq_max,
    1.2 * iq_max
)

drive_max = np.max(np.abs(drive))

ax_drive.set_ylim(
    -1.2 * drive_max,
    1.2 * drive_max
)


# =============================================================
# INITIAL TITLE
# =============================================================

final = P[-1]

ax_pop.set_title(
    "3-level transmon — DRAG   "
    f"(T = {T:.1f} ns, "
    f"A = {amplitude:.4f} GHz, "
    f"β = {beta:.2f} ns)\n"
    f"P0 = {final[0]:.5f}   "
    f"P1 = {final[1]:.5f}   "
    f"P2 = {final[2]:.3e}   "
    f"Leakage = {final[2]:.3e}"
)


# =============================================================
# SHOW
# =============================================================

plt.show()

# Lock in the values selected with the sliders
T = slider_T.val
amplitude = slider_A.val
beta = slider_beta.val

print("Final tuned parameters:")
print(f"T       = {T:.6f} ns")
print(f"Amplitude = {amplitude:.6f} GHz")
print(f"beta    = {beta:.6f}")
