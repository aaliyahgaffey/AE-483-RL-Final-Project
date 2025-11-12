import numpy as np
import mujoco
import math


class FeedforwardPitchOscillation:
    """
    Feedforward controller commanding a sinusoidal pitch oscillation
    about the y-axis (nose up/down motion).

    Desired trajectory:
        theta(t) = A * sin(omega * t)
        tau_y(t) = Iyy * theta_ddot(t)
    """

    def __init__(
        self,
        model,
        data,
        amp_deg: float = 45.0,
        n_oscillations: int = 2,
        tau_margin: float = 0.3,
        period_scale: float = 2.0,  # >1 slows oscillation down
        # --- REMOVED torque_gain ---
    ):
        self.model = model
        self.data = data

        # Identify drone body to extract inertia and mass
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.mass = model.body_mass[body_id]
        self.Iyy = model.body_inertia[body_id][1]  # pitch-axis inertia

        # Simulation step
        self.dt = float(model.opt.timestep)

        # Locate actuators: thrust and My (pitch torque)
        self.idx_thrust, self.idx_my = None, None
        for i in range(model.nu):
            gx, gy, gz, mx, my, mz = model.actuator_gear[i]
            if gz != 0 and self.idx_thrust is None:
                self.idx_thrust = i
            if my != 0 and self.idx_my is None:
                self.idx_my = i

        if self.idx_thrust is None or self.idx_my is None:
            raise RuntimeError("Thrust or My actuator not found in XML gear definitions.")

        # Oscillation setup
        self.A = math.radians(amp_deg)
        self.tau_max = float(model.actuator_ctrlrange[self.idx_my, 1])

        # Compute base omega and slow it down with period_scale
        # Check if Iyy or A are zero to avoid division error
        if self.Iyy * self.A == 0:
            self.omega = 0
        else:
            self.omega = math.sqrt(tau_margin * self.tau_max / (self.Iyy * self.A)) / period_scale
            
        if self.omega == 0:
            self.T_single = 0
        else:
            self.T_single = 2 * math.pi / self.omega
            
        self.T_total = n_oscillations * self.T_single

        # Hover thrust
        self.g = 9.81
        self.hover_thrust = self.mass * self.g

    def theta_des(self, t: float) -> float:
        """Desired pitch angle (rad)."""
        if t <= self.T_total:
            return self.A * math.sin(self.omega * t)
        else:
            # smooth decay back to zero
            return 0.0

    def theta_ddot_des(self, t: float) -> float:
        """Desired pitch angular acceleration (rad/s²)."""
        return -self.A * (self.omega ** 2) * math.sin(self.omega * t) if t <= self.T_total else 0.0

    def get_control(self, t: float):
        # Desired pitch angle and acceleration
        theta_des = self.theta_des(t)
        theta_ddot_des = self.theta_ddot_des(t)

        # --- Feedforward torque (CORRECTED) ---
        # The physically correct gain is 1.0
        tau_y = self.Iyy * theta_ddot_des
        tau_y = float(np.clip(tau_y, -self.tau_max, self.tau_max))

        # --- Feedforward thrust correction ---
        # Maintain vertical lift at nonzero pitch
        f_z = self.mass * self.g / max(np.cos(theta_des), 1e-3)
        f_z = np.clip(f_z, 0.0, self.mass * self.g * 2.0)  # safety cap

        # --- Build control vector ---
        u = np.zeros(self.model.nu)
        u[self.idx_thrust] = f_z
        u[self.idx_my] = tau_y
        return u, tau_y, f_z