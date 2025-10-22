# -----------------------------------------------------------------------------
# Project: MPC Path Tracking (BEV Extension)
# File: main_mpc_car.py
# License: MIT
# Author: Ahmad Abubakar Musa
#
# Adapted from original work by Mark Misin (© Mark Misin Engineering)
# -----------------------------------------------------------------------------

import os
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import support_files_car as sfc


def main():
    """Run Model Predictive Control + Battery Electric Vehicle (BEV) simulation."""

    print("--------------------------------------------------------")
    print("🧠  MPC + BEV Dynamics Simulation (by Ahmad A. Musa)")
    print("--------------------------------------------------------")
    print(f"Python version     : {platform.python_version()}")
    print(f"Numpy version      : {np.__version__}")
    print(f"Matplotlib version : {matplotlib.__version__}\n")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    support = sfc.SupportFilesCar()
    constants = support.constants

    Ts          = constants['Ts']
    outputs     = constants['outputs']
    hz_default  = constants['hz']
    hz          = hz_default
    time_length = constants['time_length']
    PID_switch  = constants['PID_switch']

    # -------------------------------------------------------------------------
    # Reference trajectory
    # -------------------------------------------------------------------------
    t = np.arange(0, time_length + Ts, Ts)
    r = constants['r']
    f = constants['f']
    psi_ref, X_ref, Y_ref = support.trajectory_generator(t, r, f)
    sim_length = len(t)

    # Build reference stacking vector [psi0, Y0, psi1, Y1, ...]
    refSignals = np.zeros(len(X_ref) * outputs)
    kk = 0
    for i in range(0, len(refSignals), outputs):
        refSignals[i]   = psi_ref[kk]
        refSignals[i+1] = Y_ref[kk]
        kk += 1

    # -------------------------------------------------------------------------
    # Longitudinal reference speed (for PI speed controller)
    # -------------------------------------------------------------------------
    vx_ref = constants['vx0'] * np.ones_like(t)
    # Example profile: accelerate, then cruise, then slow down
    vx_ref[t >= 2.0] = 15.0
    vx_ref[t >= 6.5] = 10.0

    # -------------------------------------------------------------------------
    # Initial states
    # -------------------------------------------------------------------------
    # Lateral states for MPC
    y_dot = 0.0
    psi = 0.0
    psi_dot = 0.0
    Y = Y_ref[0] + 10.0  # start with an offset
    states = np.array([y_dot, psi, psi_dot, Y])

    # Longitudinal & battery
    vx  = constants['vx0']
    soc = constants['soc0']

    # Logs
    statesTotal   = np.zeros((sim_length, len(states))); statesTotal[0, :] = states
    psi_opt_total = np.zeros((sim_length, hz_default))
    Y_opt_total   = np.zeros((sim_length, hz_default))

    U1 = 0.0
    UTotal = np.zeros(sim_length); UTotal[0] = U1

    Tw = 0.0
    Ttotal = np.zeros(sim_length); Ttotal[0] = Tw

    vx_log    = np.zeros(sim_length); vx_log[0] = vx
    soc_log   = np.zeros(sim_length); soc_log[0] = soc
    pelec_log = np.zeros(sim_length)

    # -------------------------------------------------------------------------
    # Helper extractors for predicted psi & Y (matching original indexing)
    # -------------------------------------------------------------------------
    def build_extractors(hz_local, nx=4, nu=1):
        C_psi_opt = np.zeros((hz_local, (nx + nu) * hz_local))
        for ii in range(1, hz_local + 1):
            C_psi_opt[ii-1][ii + nx*(ii-1)] = 1
        C_Y_opt = np.zeros((hz_local, (nx + nu) * hz_local))
        for ii in range(3, hz_local + 3):
            C_Y_opt[ii-3][ii + nx*(ii-3)] = 1
        return C_psi_opt, C_Y_opt

    C_psi_opt_full, C_Y_opt_full = build_extractors(hz_default)

    # Warm-start MPC matrices at initial speed
    Ad, Bd, Cd, Dd = support.state_space(vx)
    Hdb, Fdbt, Cdb, Adc = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)

    # PI speed control integrator
    e_int_v = 0.0
    Kp_v = constants['Kp_v']
    Ki_v = constants['Ki_v']

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    kref = 0
    for i in range(0, sim_length - 1):

        # -------------------- Lateral MPC prep --------------------
        x_aug_t = np.transpose([np.concatenate((states, [U1]), axis=0)])

        kref += outputs
        if kref + outputs*hz <= len(refSignals):
            r_vec = refSignals[kref:kref + outputs*hz]
        else:
            r_vec = refSignals[kref:len(refSignals)]
            hz -= 1  # shrink horizon near the end of the run

        # Re-linearize lateral model at current speed (LPV)
        Ad, Bd, Cd, Dd = support.state_space(vx)

        if hz < hz_default:
            Hdb, Fdbt, Cdb, Adc = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)

        # Condensed-matrix step (keeps original algebra)
        ft = np.matmul(
            np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)], r_vec), axis=0),
            Fdbt
        )
        du = -np.matmul(np.linalg.inv(Hdb), np.transpose([ft]))
        x_aug_opt = np.matmul(Cdb, du) + np.matmul(Adc, x_aug_t)

        # Predicted psi & Y for plotting
        C_psi_opt = C_psi_opt_full[0:hz, 0:((len(states) + np.size(U1)) * hz)]
        C_Y_opt   = C_Y_opt_full[0:hz,   0:((len(states) + np.size(U1)) * hz)]
        psi_opt = (C_psi_opt @ x_aug_opt).T[0]
        Y_opt   = (C_Y_opt   @ x_aug_opt).T[0]
        psi_opt_total[i+1][0:hz] = psi_opt
        Y_opt_total[i+1][0:hz]   = Y_opt

        # Apply steering increment
        U1 = U1 + du[0][0]
        U1 = np.clip(U1, -np.pi/6, np.pi/6)  # ±30°
        UTotal[i+1] = U1

        # -------------------- Longitudinal PI + Battery --------------------
        ev = vx_ref[i] - vx
        e_int_v += ev * Ts
        Tw_cmd = Kp_v * ev + Ki_v * e_int_v

        vx, soc, P_elec, Tw_limited = support.longitudinal_and_battery(vx, Tw_cmd, soc)
        vx_log[i+1]    = vx
        soc_log[i+1]   = soc
        pelec_log[i+1] = P_elec
        Ttotal[i+1]    = Tw_limited

        # -------------------- Propagate lateral states --------------------
        states = support.open_loop_new_states(states, U1, vx)
        statesTotal[i+1, :] = states

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("✅ Simulation complete.")
    print(f"Final speed: {vx:.2f} m/s | Final SOC: {soc*100:.2f}%\n")

    # -------------------------------------------------------------------------
    # Visualization (animation + static plots)
    # -------------------------------------------------------------------------
    # Save animation if env var set (used by `make anim`)
    SAVE_ANIM = os.getenv("SAVE_ANIM", "0") == "1"

    # Figure for animation
    frame_amount = int(time_length / Ts)
    lf = constants['lf']
    lr = constants['lr']
    lane_width = constants['lane_width']

    fig_x = 16; fig_y = 9
    fig = plt.figure(figsize=(fig_x, fig_y), dpi=120, facecolor=(0.8, 0.8, 0.8))
    n = 3; m = 3
    gs = gridspec.GridSpec(n, m)

    # World view
    ax0 = fig.add_subplot(gs[0,:], facecolor=(0.9, 0.9, 0.9))
    ax0.plot(X_ref, Y_ref, 'b', linewidth=1)  # Reference trajectory

    ax0.plot([X_ref[0], X_ref[frame_amount]], [ lane_width/2,  lane_width/2], 'k', linewidth=0.2)
    ax0.plot([X_ref[0], X_ref[frame_amount]], [-lane_width/2, -lane_width/2], 'k', linewidth=0.2)
    ax0.plot([X_ref[0], X_ref[frame_amount]], [ lane_width/2+lane_width,  lane_width/2+lane_width], 'k', linewidth=0.2)
    ax0.plot([X_ref[0], X_ref[frame_amount]], [-lane_width/2-lane_width, -lane_width/2-lane_width], 'k', linewidth=0.2)
    ax0.plot([X_ref[0], X_ref[frame_amount]], [ lane_width/2+2*lane_width,  lane_width/2+2*lane_width], 'k', linewidth=3)
    ax0.plot([X_ref[0], X_ref[frame_amount]], [-lane_width/2-2*lane_width, -lane_width/2-2*lane_width], 'k', linewidth=3)

    (car_1,)          = ax0.plot([], [], 'k', linewidth=3)
    (car_predicted,)  = ax0.plot([], [], '-m', linewidth=1)
    (car_determined,) = ax0.plot([], [], '-r', linewidth=1)

    # Updated attribution
    ax0.text(0, 20, '© Ahmad (Adapted from Mark Misin Engineering)', size=15)

    ax0.set_xlim(X_ref[0], X_ref[frame_amount])
    ax0.set_ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2), X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
    ax0.set_ylabel('Y-distance [m]', fontsize=15)

    # Zoomed vehicle view
    ax1 = fig.add_subplot(gs[1,:], facecolor=(0.9, 0.9, 0.9))
    bbox_props_angle = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='k', lw=1.0)
    bbox_props_steer = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='r', lw=1.0)

    ax1.plot([-50, 50], [0, 0], 'k', linewidth=1)
    (car_1_body,)             = ax1.plot([], [], 'k', linewidth=3)
    (car_1_body_extension,)   = ax1.plot([], [], '--k', linewidth=1)
    (car_1_back_wheel,)       = ax1.plot([], [], 'r', linewidth=4)
    (car_1_front_wheel,)      = ax1.plot([], [], 'r', linewidth=4)
    (car_1_front_wheel_ext,)  = ax1.plot([], [], '--r', linewidth=1)

    n1_start = -5; n1_finish = 30
    ax1.set_xlim(n1_start, n1_finish)
    ax1.set_ylim(-(n1_finish - n1_start)/(n*(fig_x/fig_y)*2),
                  (n1_finish - n1_start)/(n*(fig_x/fig_y)*2))
    ax1.set_ylabel('Y-distance [m]', fontsize=15)
    yaw_angle_text = ax1.text(25,  2.0, '', size='20', color='k', bbox=bbox_props_angle)
    steer_angle    = ax1.text(25, -2.5, '', size='20', color='r', bbox=bbox_props_steer)

    # Time histories
    ax2 = fig.add_subplot(gs[2,0], facecolor=(0.9, 0.9, 0.9))
    (steering_wheel,) = ax2.plot([], [], '-r', linewidth=1, label='steering angle [rad]')
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(np.min(UTotal)-0.1, np.max(UTotal)+0.1)
    ax2.set_xlabel('time [s]', fontsize=15)
    ax2.grid(True); ax2.legend(loc='upper right', fontsize='small')

    ax3 = fig.add_subplot(gs[2,1], facecolor=(0.9, 0.9, 0.9))
    ax3.plot(t, psi_ref, '-b', linewidth=1, label='yaw reference [rad]')
    (yaw_angle_line,) = ax3.plot([], [], '-r', linewidth=1, label='yaw angle [rad]')
    (psi_pred_line,)  = ax3.plot([], [], '-m', linewidth=3, label='psi - predicted [rad]')
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(np.min(statesTotal[:,1])-0.1, np.max(statesTotal[:,1])+0.1)
    ax3.set_xlabel('time [s]', fontsize=15)
    ax3.grid(True); ax3.legend(loc='upper right', fontsize='small')

    ax4 = fig.add_subplot(gs[2,2], facecolor=(0.9, 0.9, 0.9))
    ax4.plot(t, Y_ref, '-b', linewidth=1, label='Y - reference [m]')
    (Y_pos_line,)   = ax4.plot([], [], '-r', linewidth=1, label='Y - position [m]')
    (Y_pred_line,)  = ax4.plot([], [], '-m', linewidth=3, label='Y - predicted [m]')
    ax4.set_xlim(0, t[-1])
    ax4.set_ylim(np.min(statesTotal[:,3])-2, np.max(statesTotal[:,3])+2)
    ax4.set_xlabel('time [s]', fontsize=15)
    ax4.grid(True); ax4.legend(loc='upper right', fontsize='small')

    # Update function for animation
    def update_plot(num):
        hz_local = min(hz_default, len(t) - num)

        # Car pose in world view
        car_1.set_data(
            [X_ref[num] - lr*np.cos(statesTotal[num,1]), X_ref[num] + lf*np.cos(statesTotal[num,1])],
            [statesTotal[num,3] - lr*np.sin(statesTotal[num,1]), statesTotal[num,3] + lf*np.sin(statesTotal[num,1])]
        )

        # Zoomed geometry
        car_1_body.set_data(
            [-lr*np.cos(statesTotal[num,1]), lf*np.cos(statesTotal[num,1])],
            [-lr*np.sin(statesTotal[num,1]), lf*np.sin(statesTotal[num,1])]
        )
        car_1_body_extension.set_data(
            [0, (lf+40)*np.cos(statesTotal[num,1])],
            [0, (lf+40)*np.sin(statesTotal[num,1])]
        )
        car_1_back_wheel.set_data(
            [-(lr+0.5)*np.cos(statesTotal[num,1]), -(lr-0.5)*np.cos(statesTotal[num,1])],
            [-(lr+0.5)*np.sin(statesTotal[num,1]), -(lr-0.5)*np.sin(statesTotal[num,1])]
        )
        car_1_front_wheel.set_data(
            [lf*np.cos(statesTotal[num,1]) - 0.5*np.cos(statesTotal[num,1]+UTotal[num]),
             lf*np.cos(statesTotal[num,1]) + 0.5*np.cos(statesTotal[num,1]+UTotal[num])],
            [lf*np.sin(statesTotal[num,1]) - 0.5*np.sin(statesTotal[num,1]+UTotal[num]),
             lf*np.sin(statesTotal[num,1]) + 0.5*np.sin(statesTotal[num,1]+UTotal[num])]
        )
        car_1_front_wheel_ext.set_data(
            [lf*np.cos(statesTotal[num,1]), lf*np.cos(statesTotal[num,1]) + (0.5+40)*np.cos(statesTotal[num,1]+UTotal[num])],
            [lf*np.sin(statesTotal[num,1]), lf*np.sin(statesTotal[num,1]) + (0.5+40)*np.sin(statesTotal[num,1]+UTotal[num])]
        )

        yaw_angle_text.set_text(str(round(statesTotal[num,1],2)) + ' rad')
        steer_angle.set_text(str(round(UTotal[num],2)) + ' rad')

        steering_wheel.set_data(t[0:num], UTotal[0:num])
        yaw_angle_line.set_data(t[0:num], statesTotal[0:num,1])
        Y_pos_line.set_data(t[0:num], statesTotal[0:num,3])

        if PID_switch != 1 and num != 0:
            Y_pred_line.set_data(t[num:num+hz_local],  Y_opt_total[num][0:hz_local])
            psi_pred_line.set_data(t[num:num+hz_local], psi_opt_total[num][0:hz_local])
            car_predicted.set_data(X_ref[num:num+hz_local], Y_opt_total[num][0:hz_local])

        car_determined.set_data(X_ref[0:num], statesTotal[0:num,3])

        if PID_switch != 1:
            return (car_1, car_1_body, car_1_body_extension, car_1_back_wheel,
                    car_1_front_wheel, car_1_front_wheel_ext, yaw_angle_text,
                    steer_angle, steering_wheel, yaw_angle_line, Y_pos_line,
                    car_determined, Y_pred_line, psi_pred_line, car_predicted)
        else:
            return (car_1, car_1_body, car_1_body_extension, car_1_back_wheel,
                    car_1_front_wheel, car_1_front_wheel_ext, yaw_angle_text,
                    steer_angle, steering_wheel, yaw_angle_line, Y_pos_line,
                    car_determined)

    # Create animation
    car_ani = animation.FuncAnimation(
        fig, update_plot, frames=frame_amount, interval=20, repeat=True, blit=True
    )

    # Save or show
    if SAVE_ANIM:
        print("💾 Saving animation to 'mpc_bev_demo.mp4' ...")
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata={'artist': 'Ahmad A. Musa'}, bitrate=2400)
            car_ani.save('mpc_bev_demo.mp4', writer=writer)
            print("✅ Saved: mpc_bev_demo.mp4")
        except Exception as e:
            print("⚠️ Could not save MP4 via ffmpeg writer:", e)
    else:
        plt.show()

    # ------------------------ Static plots ------------------------
    # World trajectory
    plt.figure()
    plt.plot(X_ref, Y_ref, 'b', linewidth=2, label='Reference trajectory')
    plt.plot(X_ref, statesTotal[:,3], '--r', linewidth=2, label='Car position')
    plt.xlabel('x-position [m]'); plt.ylabel('y-position [m]')
    plt.grid(True); plt.legend(loc='upper right')
    plt.title("Vehicle Path Tracking (MPC vs Reference)")
    plt.ylim(-X_ref[-1]/2, X_ref[-1]/2)
    plt.show()

    # Inputs & outputs
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, UTotal[:], 'r', linewidth=2, label='steering [rad]')
    plt.ylabel('δ [rad]')
    plt.grid(True); plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, psi_ref, 'b', linewidth=2, label='yaw_ref')
    plt.plot(t, statesTotal[:,1], '--r', linewidth=2, label='yaw')
    plt.ylabel('ψ [rad]')
    plt.grid(True); plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t, Y_ref, 'b', linewidth=2, label='Y_ref')
    plt.plot(t, statesTotal[:,3], '--r', linewidth=2, label='Y')
    plt.xlabel('time [s]'); plt.ylabel('Y [m]')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

    # Speed tracking
    plt.figure()
    plt.plot(t, vx_ref, label='v_ref [m/s]')
    plt.plot(t, vx_log, '--', label='v_actual [m/s]')
    plt.xlabel('time [s]'); plt.ylabel('speed [m/s]')
    plt.title("Longitudinal Speed Tracking")
    plt.grid(True); plt.legend()
    plt.show()

    # Battery SOC
    plt.figure()
    plt.plot(t, soc_log, label='SOC [-]')
    plt.xlabel('time [s]'); plt.ylabel('state of charge [-]')
    plt.title("Battery SOC over Time")
    plt.grid(True); plt.legend()
    plt.show()

    # Electrical power
    plt.figure()
    plt.plot(t, pelec_log/1000.0, label='Power [kW]')
    plt.xlabel('time [s]'); plt.ylabel('Electrical Power [kW]')
    plt.title("Battery Power Profile")
    plt.grid(True); plt.legend()
    plt.show()


# -----------------------------------------------------------------------------
# Entry point for command-line execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
