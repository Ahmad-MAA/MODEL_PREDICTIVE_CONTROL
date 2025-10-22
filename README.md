
# 🧠 MPC-BEV — Model Predictive Control for Path Tracking with Battery-Electric Vehicle Dynamics

A complete Python simulation of **Model Predictive Control (MPC)** for autonomous path tracking, extended with a simplified **Battery Electric Vehicle (BEV)** model.  
The simulator combines *lateral MPC steering control* with *longitudinal PI torque control* and a *battery state-of-charge (SOC)* model.

> ✨ Developed and maintained by **Ahmad Abubakar Musa**  
> 🧩 Adapted from original work by **Mark Misin Engineering**

---

## 🚗 Overview

This project demonstrates how MPC can guide a vehicle along a reference trajectory while accounting for real-world BEV dynamics.

**Key capabilities**

- ⚙️ **Lateral control (MPC)** – minimizes yaw & lateral position error  
- 🔋 **Longitudinal control (PI)** – tracks a speed reference via wheel torque  
- ⚡ **BEV powertrain model** – aerodynamic drag, rolling resistance, torque limit, SOC & regenerative energy  
- 🎥 **Animation & plots** – visualize vehicle motion, control inputs, speed, battery SOC, and power profile  

---

## 🧩 File Structure

```

mpc_bev_sim/
├── main_mpc_car.py          # Main script – runs MPC simulation & animation
├── support_files_car.py     # Support module – dynamics, MPC matrices, battery model
├── **init**.py              # Package initializer
├── README.md
├── LICENSE
├── requirements.txt
└── pyproject.toml

````

---

## ⚙️ Installation

Clone and install locally (editable mode recommended):

```bash
git clone https://github.com/ahmad-musa/mpc-bev-sim.git
cd mpc-bev-sim
pip install -e .
````

> 🧰 Requirements: Python ≥ 3.8, NumPy ≥ 1.20, Matplotlib ≥ 3.3
> (optional) For MP4 animation export, install `imageio-ffmpeg` or system `ffmpeg`.

---

## ▶️ Running the Simulation

### Option 1 — via console command

(available after installation)

```bash
mpc-bev-demo
```

### Option 2 — directly with Python

```bash
python main_mpc_car.py
```

You’ll see:

* an animated car following a curved lane,
* steering, yaw, and lateral plots,
* BEV metrics: **speed**, **SOC**, and **power**.

### To save animation as MP4

```bash
SAVE_ANIM=1 MPLBACKEND=Agg python main_mpc_car.py
```

> This creates `mpc_bev_demo.mp4` in the project folder.

---

## 🔋 BEV & Control Model

**Longitudinal Dynamics**
[
m\dot{v}_x = \frac{T_w}{R_w} - \frac{1}{2}\rho C_d A v_x^2 - m g C_r
]

**Battery SOC**
[
SOC_{k+1} = SOC_k - \frac{P_{elec},T_s}{E_{batt,max}}
]

**Power flow**
[
P_{elec} =
\begin{cases}
\dfrac{T_w v_x}{R_w \eta_{drive}}, & P_{mech} > 0 \
T_w v_x \eta_{regen}/R_w, & P_{mech} < 0
\end{cases}
]

**Lateral Dynamics (bicycle model, LPV in (v_x))**
[
\dot{x} = A(v_x)x + B(v_x)\delta
]

The matrices are discretized every step and fed to the MPC optimizer.

---

## 📊 Output Visualizations

* **Trajectory tracking** — MPC vs. reference path
* **Steering angle** — control effort over time
* **Yaw angle & lateral position** — closed-loop accuracy
* **Speed tracking** — longitudinal PI control
* **Battery SOC** — energy consumption or regeneration
* **Electrical power** — drive/regen profile in kW

---

## 🪪 License & Attribution

This project is released under the **MIT License** (see `LICENSE`).

**Adaptations & BEV extensions** — © 2025 Ahmad Abubakar Musa
**Original control structure & concepts** — © Mark Misin Engineering

If you reuse or extend this work, please retain both names in headers or citations.

---

## 💡 Notes

* Educational simulator for MPC + BEV fundamentals.
* Simplified single-track (bicycle) model; not for real-time control use.
* Replace constants in `support_files_car.py` to tune mass, battery, or control gains.

---

### ✉️ Author

**Ahmad Abubakar Musa**
Renewable Energy & Control Systems Engineer
📍 Glasgow, United Kingdom
📧 [ahmad@example.com](mailto:ahmad@example.com)

---

Enjoy exploring the synergy between **control systems**, **vehicle dynamics**, and **energy management** ⚡🚘



