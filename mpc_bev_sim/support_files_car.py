# -----------------------------------------------------------------------------
# Project: MPC Path Tracking (BEV Extension)
# File: support_files_car.py
# License: MIT
# Author: Ahmad Abubakar Musa
#
# Adapted from original work by Mark Misin (© Mark Misin Engineering)
# -----------------------------------------------------------------------------

import numpy as np


class SupportFilesCar:
    """
    Support functions for the BEV-based MPC simulation.
    Provides constants, state-space matrices, and physical models.
    """

    def __init__(self):
        """Initialize constants for both vehicle dynamics and BEV parameters."""

        # ---------------------- Vehicle parameters ----------------------
        m = 1500.0          # Vehicle mass [kg]
        Iz = 3000.0         # Yaw moment of inertia [kg·m²]
        Caf = 19000.0       # Front tire cornering stiffness [N/rad]
        Car = 33000.0       # Rear tire cornering stiffness [N/rad]
        lf = 2.0            # Distance from CG to front axle [m]
        lr = 3.0            # Distance from CG to rear axle [m]
        Ts = 0.02           # Sampling time [s]

        # ---------------------- MPC tuning ----------------------
        Q = np.matrix('1 0; 0 1')   # Output weights
        S = np.matrix('1 0; 0 1')   # Final horizon weights
        R = np.matrix('1')          # Input weight
        outputs = 2
        hz = 20                     # Prediction horizon

        # ---------------------- Path & simulation ----------------------
        x_dot = 20.0                # initial forward speed [m/s]
        lane_width = 7.0            # [m]
        r = 4.0                     # amplitude for sinusoidal trajectory
        f = 0.01                    # frequency
        time_length = 10.0          # total sim time [s]

        # ---------------------- PID fallback (not used in MPC) ----------------------
        PID_switch = 0
        Kp_yaw = 7; Kd_yaw = 3; Ki_yaw = 5
        Kp_Y = 7; Kd_Y = 3; Ki_Y = 5

        # ---------------------- BEV / Longitudinal parameters ----------------------
        vx0 = 10.0                  # initial longitudinal velocity [m/s]
        soc0 = 1.0                  # normalized initial SOC [1 = full]
        battery_capacity = 60_000.0 * 3600.0  # 60 kWh in Joules
        eta_drive = 0.92            # drivetrain efficiency
        wheel_radius = 0.3          # [m]
        max_torque = 400.0          # [Nm]
        drag_coeff = 0.32           # drag coefficient
        air_density = 1.225         # [kg/m³]
        frontal_area = 2.2          # [m²]
        Crr = 0.015                 # rolling resistance coefficient
        Kp_v = 250.0                # PI gains for speed controller
        Ki_v = 40.0

        # ---------------------- Miscellaneous ----------------------
        trajectory = 3

        # Bundle constants
        self.constants = {
            'm': m, 'Iz': Iz, 'Caf': Caf, 'Car': Car, 'lf': lf, 'lr': lr,
            'Ts': Ts, 'Q': Q, 'S': S, 'R': R,
            'outputs': outputs, 'hz': hz,
            'x_dot': x_dot, 'r': r, 'f': f,
            'time_length': time_length,
            'lane_width': lane_width,
            'PID_switch': PID_switch,
            'Kp_yaw': Kp_yaw, 'Kd_yaw': Kd_yaw, 'Ki_yaw': Ki_yaw,
            'Kp_Y': Kp_Y, 'Kd_Y': Kd_Y, 'Ki_Y': Ki_Y,
            'trajectory': trajectory,
            'vx0': vx0, 'soc0': soc0, 'battery_capacity': battery_capacity,
            'eta_drive': eta_drive, 'wheel_radius': wheel_radius,
            'max_torque': max_torque,
            'drag_coeff': drag_coeff, 'air_density': air_density,
            'frontal_area': frontal_area, 'Crr': Crr,
            'Kp_v': Kp_v, 'Ki_v': Ki_v,
        }

    # -------------------------------------------------------------------------
    # Trajectory generation
    # -------------------------------------------------------------------------
    def trajectory_generator(self, t, r, f):
        """Generate a reference trajectory for the vehicle to follow."""

        Ts = self.constants['Ts']
        x_dot = self.constants['x_dot']
        trajectory = self.constants['trajectory']

        x = np.linspace(0, x_dot * t[-1], num=len(t))

        if trajectory == 1:
            y = -9 * np.ones(len(t))
        elif trajectory == 2:
            y = 9 * np.tanh(t - t[-1] / 2)
        elif trajectory == 3:
            aaa = -28 / 100**2 / 1.1
            bbb = -14 if aaa > 0 else 14
            y_1 = aaa * (x + self.constants['lane_width'] - 100)**2 + bbb
            y_2 = 2 * r * np.sin(2 * np.pi * f * x)
            y = (y_1 + y_2) / 2
        else:
            raise ValueError("Trajectory must be 1, 2, or 3")

        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])

        psi = np.arctan2(dy, dx)
        psiInt = np.zeros_like(psi)
        psiInt[0] = psi[0]
        dpsi = np.diff(psi, prepend=psi[0])

        for i in range(1, len(psiInt)):
            if dpsi[i - 1] < -np.pi:
                psiInt[i] = psiInt[i - 1] + (dpsi[i - 1] + 2 * np.pi)
            elif dpsi[i - 1] > np.pi:
                psiInt[i] = psiInt[i - 1] + (dpsi[i - 1] - 2 * np.pi)
            else:
                psiInt[i] = psiInt[i - 1] + dpsi[i - 1]

        return psiInt, x, y

    # -------------------------------------------------------------------------
    # State-space representation (LPV on vx)
    # -------------------------------------------------------------------------
    def state_space(self, vx):
        """Return discrete state-space matrices for lateral dynamics, parameterized by vx."""
        m = self.constants['m']
        Iz = self.constants['Iz']
        Caf = self.constants['Caf']
        Car = self.constants['Car']
        lf = self.constants['lf']
        lr = self.constants['lr']
        Ts = self.constants['Ts']

        if vx < 0.1:
            vx = 0.1  # avoid divide-by-zero

        A1 = -(2*Caf + 2*Car) / (m * vx)
        A2 = -vx - (2*Caf*lf - 2*Car*lr) / (m * vx)
        A3 = -(2*lf*Caf - 2*lr*Car) / (Iz * vx)
        A4 = -(2*lf**2*Caf + 2*lr**2*Car) / (Iz * vx)

        A = np.array([[A1, 0, A2, 0],
                      [0, 0, 1, 0],
                      [A3, 0, A4, 0],
                      [1, vx, 0, 0]])

        B = np.array([[2*Caf/m],
                      [0],
                      [2*lf*Caf/Iz],
                      [0]])

        C = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 1]])
        D = 0

        # Discretization (Euler)
        Ad = np.identity(A.shape[1]) + Ts * A
        Bd = Ts * B
        Cd = C
        Dd = D

        return Ad, Bd, Cd, Dd

    # -------------------------------------------------------------------------
    # MPC simplification matrices (same as original)
    # -------------------------------------------------------------------------
    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz):
        """Compact matrices for MPC quadratic cost minimization."""
        A_aug = np.concatenate((Ad, Bd), axis=1)
        temp = np.concatenate((np.zeros((Bd.shape[1], Ad.shape[1])),
                               np.eye(Bd.shape[1])), axis=1)
        A_aug = np.concatenate((A_aug, temp), axis=0)
        B_aug = np.concatenate((Bd, np.eye(Bd.shape[1])), axis=0)
        C_aug = np.concatenate((Cd, np.zeros((Cd.shape[0], Bd.shape[1]))), axis=1)

        Q = self.constants['Q']; S = self.constants['S']; R = self.constants['R']

        CQC = C_aug.T @ Q @ C_aug
        CSC = C_aug.T @ S @ C_aug
        QC  = Q @ C_aug
        SC  = S @ C_aug

        Qdb = np.zeros((CQC.shape[0]*hz, CQC.shape[1]*hz))
        Tdb = np.zeros((QC.shape[0]*hz,  QC.shape[1]*hz))
        Rdb = np.zeros((R.shape[0]*hz,   R.shape[1]*hz))
        Cdb = np.zeros((B_aug.shape[0]*hz, B_aug.shape[1]*hz))
        Adc = np.zeros((A_aug.shape[0]*hz, A_aug.shape[1]))

        for i in range(hz):
            if i == hz - 1:
                Qdb[i*CQC.shape[0]:(i+1)*CQC.shape[0], i*CQC.shape[1]:(i+1)*CQC.shape[1]] = CSC
                Tdb[i*QC.shape[0]:(i+1)*QC.shape[0],   i*QC.shape[1]:(i+1)*QC.shape[1]]   = SC
            else:
                Qdb[i*CQC.shape[0]:(i+1)*CQC.shape[0], i*CQC.shape[1]:(i+1)*CQC.shape[1]] = CQC
                Tdb[i*QC.shape[0]:(i+1)*QC.shape[0],   i*QC.shape[1]:(i+1)*QC.shape[1]]   = QC

            Rdb[i*R.shape[0]:(i+1)*R.shape[0], i*R.shape[1]:(i+1)*R.shape[1]] = R

            for j in range(hz):
                if j <= i:
                    Cdb[i*B_aug.shape[0]:(i+1)*B_aug.shape[0],
                         j*B_aug.shape[1]:(j+1)*B_aug.shape[1]] = \
                         np.linalg.matrix_power(A_aug, (i - j)) @ B_aug

            Adc[i*A_aug.shape[0]:(i+1)*A_aug.shape[0], :] = np.linalg.matrix_power(A_aug, i+1)

        Hdb  = Cdb.T @ Qdb @ Cdb + Rdb
        Fdbt = np.vstack((-Adc.T @ Qdb @ Cdb, -Tdb @ Cdb))

        return Hdb, Fdbt, Cdb, Adc

    # -------------------------------------------------------------------------
    # Lateral open-loop integration (uses current vx)
    # -------------------------------------------------------------------------
    def open_loop_new_states(self, states, U1, vx):
        """Propagate lateral states over one Ts using given steering and longitudinal speed."""
        m = self.constants['m']; Iz = self.constants['Iz']
        Caf = self.constants['Caf']; Car = self.constants['Car']
        lf = self.constants['lf']; lr = self.constants['lr']
        Ts = self.constants['Ts']

        y_dot, psi, psi_dot, Y = states
        sub_loop = 30

        for _ in range(sub_loop):
            y_dot_dot = (-(2*Caf + 2*Car)/(m*vx))*y_dot \
                        + (-vx - (2*Caf*lf - 2*Car*lr)/(m*vx))*psi_dot \
                        + 2*Caf/m * U1

            psi_dot_dot = (-(2*lf*Caf - 2*lr*Car)/(Iz*vx))*y_dot \
                          - ((2*lf**2*Caf + 2*lr**2*Car)/(Iz*vx))*psi_dot \
                          + 2*lf*Caf/Iz * U1

            Y_dot = np.sin(psi)*vx + np.cos(psi)*y_dot

            y_dot += y_dot_dot * Ts / sub_loop
            psi += psi_dot * Ts / sub_loop
            psi_dot += psi_dot_dot * Ts / sub_loop
            Y += Y_dot * Ts / sub_loop

        return np.array([y_dot, psi, psi_dot, Y])

    # -------------------------------------------------------------------------
    # Longitudinal & battery subsystem
    # -------------------------------------------------------------------------
    def longitudinal_and_battery(self, vx, Tw_cmd, soc):
        """
        Update longitudinal speed and battery SOC using wheel torque command.
        Returns: new_vx, new_soc, P_elec, Tw_limited
        """

        const = self.constants
        Ts = const['Ts']
        m = const['m']
        r_w = const['wheel_radius']
        eta = const['eta_drive']
        rho = const['air_density']
        Cd = const['drag_coeff']
        A = const['frontal_area']
        Crr = const['Crr']
        Tw_max = const['max_torque']
        Ebatt = const['battery_capacity']

        Tw = np.clip(Tw_cmd, -Tw_max, Tw_max)

        F_aero = 0.5 * rho * Cd * A * vx**2
        F_roll = m * 9.81 * Crr
        Fx = Tw / r_w
        a = (Fx - F_aero - F_roll) / m

        vx_new = max(0.0, vx + a * Ts)

        # Electrical power (positive for discharge)
        P_mech = Tw * vx / r_w
        P_elec = P_mech / eta if P_mech >= 0 else P_mech * eta

        # Update SOC
        soc_new = soc - (P_elec * Ts) / Ebatt
        soc_new = np.clip(soc_new, 0.0, 1.0)

        return vx_new, soc_new, P_elec, Tw


# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
