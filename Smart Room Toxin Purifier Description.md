# 💨 Smart-Room Toxin Purifier ✨

Welcome to your clean air era. POV: you need to visualize continuous control systems and 3D calculus, but you also want it to look *aesthetic*. 

This is a 3D simulation of a smart room dealing with toxic leaks. It uses a **Laplace-domain PID controller** to adjust a purifier fan based on the total toxin mass in the room, calculated via **triple integration**. Oh, and the diffusion physics? Solved with a NumPy PDE solver so it runs at 33 physics updates per second without lagging your PC. It understood the assignment.

---

## 💅 The Flex (Features)

* **NumPy PDE Solver:** Calculating 3D diffusion 50x faster than standard Python loops. It's giving maximum efficiency.
* **Laplace-Bilinear PID:** A discrete Tustin PID controller keeping the fan speed in check, complete with anti-windup clamping so the math doesn't lose the plot.
* **Interactive 3D Visuals:** Built with VPython. Featuring particle smoke, shockwave rings when a leak drops, 3D flow-field arrows, and a room that literally pulses red when the danger level peaks.
* **Live Tuning:** Adjust the PID gains in real-time with sliders to see how the system reacts to different control strategies.

---

## 🤓 The Math Behind the Magic

We are literally doing calculus in real-time. No jumpscares, just pure logic.

### 1. 3D Concentration Field (PDE)
The spread of toxins is governed by the diffusion equation. We use a 6-neighbor finite difference Laplacian in NumPy to solve this:
$$\frac{\partial C}{\partial t} = D \cdot \nabla^2 C + \sum \text{Source}_k(x,y,z) - \text{Sink}(x,y,z)$$

### 2. Triple Integration
We find the total toxin load $T(t)$ by integrating the concentration $C$ over the entire room volume $V$. NumPy handles the sum over our 3D grid instantly.

### 3. Continuous to Discrete PID
The fan speed is determined by a continuous Laplace-domain controller, mapped to discrete time using the Bilinear (Tustin) transform to prevent integrator windup:
$$\frac{U(s)}{E(s)} = K_p + \frac{K_i}{s} + \frac{K_d \cdot Ns}{s+N}$$

---

## 🎮 How to Play (Controls)

* **`[ T ]`** : Trigger a toxic leak (up to 3 active at once). Cue the shockwaves and particle smoke.
* **`[ R ]`** : Reset the room. Clear the air and reset the integrator back to zero.
* **Sliders** : Tweak the $K_p$, $K_i$, and $K_d$ variables at the bottom of the screen. Try setting $K_p$ way too high and watch the fan controller go absolutely feral.

---

## 🚀 Quickstart

No gatekeeping here. To run this on your local machine, you just need Python and two libraries.

### Prerequisites
Create a `requirements.txt` file in your folder and drop this in:
```text
vpython==7.6.4
numpy>=1.24.0
```

### Installation & Run
Pop open your terminal and run these commands:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Ouaelkk/Smart-Room-Purifier.git
   cd smart-toxin-purifier
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the sim:**
   ```bash
   python smart_room_purifier.py
   ```

A browser window will automatically open with the 3D scene. Vibe with the data and watch the live graphs do their thing.

---

## ✨ Serving Code (How to Contribute)

We love a collaborative moment. If you want to add new features, fix bugs, or just make the math more elegant, here is the vibe check for contributing:

1. **Fork it:** Make your own copy of the repo.
2. **Branch out:** Create a new branch for your feature (`git checkout -b feature/aesthetic-lasers`).
3. **Spit your facts:** Write your code. Make sure it doesn't break the NumPy PDE solver.
4. **Commit:** Write a commit message that actually makes sense. 
5. **Push & PR:** Push to your branch and open a Pull Request. 

**What we are looking for:**
* Better 3D visual effects or particle systems.
* Optimization of the integration loops.
* More chaos (like new types of toxin leaks or airflow dynamics).

Respect the vibe, keep the code clean, and we'll merge it.

---
