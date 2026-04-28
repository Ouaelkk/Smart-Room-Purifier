from vpython import *
import numpy as np
import random, math

# ================================================================
#  SMART-ROOM TOXIN PURIFIER  v3
#  ─────────────────────────────────────────────────────────────
#  MATH:
#    · 3-D Concentration Field   C(x,y,z,t)
#      ∂C/∂t = D·∇²C + Σ Source_k(x,y,z) − Sink(x,y,z)
#    · Triple Integration        T(t) = ∭_V C dV   (NumPy)
#    · Laplace-Domain PID        U(s)/E(s) = Kp + Ki/s + Kd·Ns/(s+N)
#      → Bilinear (Tustin) discretisation, anti-windup clamp
#
#  VISUAL / INTERACTIVITY:
#    · NumPy PDE solver  (50× faster than Python loops)
#    · Precomputed Gaussian source & sink masks
#    · Up to 3 simultaneous leak sources   [ T ]
#    · Shock-wave ring burst on each leak
#    · 3-D flow-field arrows (advection visualisation)
#    · Particle smoke with attach_trail tails
#    · Danger-pulse flashing room walls
#    · Live Kp / Ki / Kd sliders in caption
#    · Reset                               [ R ]
# ================================================================

# ── Scene ────────────────────────────────────────────────────────
scene = canvas(
    title=(
        "Smart-Room Toxin Purifier  v3  |  "
        "NumPy PDE · Laplace-Bilinear PID · Triple Integration"
    ),
    width=1000, height=530,
    background=color.black
)
scene.camera.pos  = vector(0, 9, 22)
scene.camera.axis = vector(0, -2, -8)
scene.lights = []
distant_light(direction=vector( 0.3, 1.0,  0.5), color=color.white * 0.45)
distant_light(direction=vector(-1.0,-0.5, -1.0), color=color.cyan  * 0.15)

# ── Room ─────────────────────────────────────────────────────────
RW, RH, RD = 12, 8, 12

room = box(
    pos=vector(0, 0, 0), size=vector(RW, RH, RD),
    color=color.white, opacity=0.06
)
box(
    pos=vector(0, -4.1, 0), size=vector(RW+0.5, 0.2, RD+0.5),
    color=color.gray(0.20)
)

# Ceiling laser-grid (decorative)
for i in range(-5, 6, 2):
    cylinder(pos=vector(i,  4, -6), axis=vector( 0, 0, 12),
             radius=0.013, color=color.cyan, opacity=0.20, emissive=True)
    cylinder(pos=vector(-6, 4,  i), axis=vector(12, 0,  0),
             radius=0.013, color=color.cyan, opacity=0.20, emissive=True)

# Corner pillars
for cx, cz in [(-5.8, -5.8), (-5.8, 5.8), (5.8, -5.8), (5.8, 5.8)]:
    box(pos=vector(cx, 0, cz), size=vector(0.34, 8.2, 0.34),
        color=color.gray(0.28))

# ── Purifier Unit ────────────────────────────────────────────────
P_BASE   = vector(0, -4, 0)
BLADE_HUB = P_BASE + vector(0, 3.1, 0)

pur_body = cylinder(
    pos=P_BASE, axis=vector(0, 3.2, 0),
    radius=1.35, color=color.cyan, emissive=True, opacity=0.88
)
pur_dome = sphere(
    pos=BLADE_HUB + vector(0, 0.1, 0),
    radius=0.60, color=color.white, emissive=True
)
pur_glow = sphere(
    pos=P_BASE + vector(0, 1.6, 0),
    radius=1.55, color=color.cyan, opacity=0.04, emissive=True
)

pur_rings = []
for yoff in [0.75, 1.55, 2.35]:
    r = ring(
        pos=P_BASE + vector(0, yoff, 0), axis=vector(0, 1, 0),
        radius=1.65, thickness=0.065,
        color=color.cyan, emissive=True, opacity=0.0
    )
    pur_rings.append(r)

fan_blades = []
for ang_deg in range(0, 360, 90):
    rad = math.radians(ang_deg)
    fb  = box(
        pos=BLADE_HUB + vector(math.cos(rad)*1.1, 0, math.sin(rad)*1.1),
        size=vector(1.55, 0.055, 0.42),
        color=color.cyan, opacity=0.90, emissive=True
    )
    fan_blades.append(fb)

# ── 3-D Concentration Grid (NumPy) ───────────────────────────────
NX, NY, NZ = 7, 5, 7
dx, dy, dz  = RW/NX, RH/NY, RD/NZ
dV          = dx * dy * dz
D_diff      = 0.35                   # diffusion coefficient [m²/s]

C = np.zeros((NX, NY, NZ))          # concentration field

# Cell-centre world coordinates
cell_x = np.linspace(-RW/2 + dx/2,  RW/2 - dx/2, NX)
cell_y = np.linspace(-RH/2 + dy/2,  RH/2 - dy/2, NY)
cell_z = np.linspace(-RD/2 + dz/2,  RD/2 - dz/2, NZ)

def cell_world(i, j, k):
    return vector(float(cell_x[i]), float(cell_y[j]), float(cell_z[k]))

# ── Gaussian mask (vectorised, no loops) ─────────────────────────
def gaussian_mask(gi, gj, gk, sigma):
    ii = np.arange(NX).reshape(NX, 1, 1)
    jj = np.arange(NY).reshape(1, NY, 1)
    kk = np.arange(NZ).reshape(1, 1, NZ)
    return np.exp(-((ii-gi)**2 + (jj-gj)**2 + (kk-gk)**2) / sigma)

# Fixed sink mask — purifier at floor centre, precomputed once
PUR_GI    = (NX//2, 0, NZ//2)
sink_mask = gaussian_mask(*PUR_GI, sigma=4.5) * 10.0   # shape (NX,NY,NZ)

# ── NumPy Laplacian  ∇²C  (6-neighbour finite differences) ──────
def laplacian_np(C):
    lap = np.zeros_like(C)
    # x — interior + Neumann boundaries (zero-flux)
    lap[1:-1, :, :] += (C[2:,:,:] + C[:-2,:,:] - 2*C[1:-1,:,:]) / dx**2
    lap[0,    :, :] += (C[1,:,:]  - C[0,:,:])   / dx**2
    lap[-1,   :, :] += (C[-2,:,:] - C[-1,:,:])  / dx**2
    # y
    lap[:, 1:-1, :] += (C[:,2:,:] + C[:,:-2,:] - 2*C[:,1:-1,:]) / dy**2
    lap[:, 0,    :] += (C[:,1,:]  - C[:,0,:])   / dy**2
    lap[:, -1,   :] += (C[:,-2,:] - C[:,-1,:])  / dy**2
    # z
    lap[:, :, 1:-1] += (C[:,:,2:] + C[:,:,:-2] - 2*C[:,:,1:-1]) / dz**2
    lap[:, :, 0   ] += (C[:,:,1]  - C[:,:,0])   / dz**2
    lap[:, :, -1  ] += (C[:,:,-2] - C[:,:,-1])  / dz**2
    return lap

# Voxel spheres — one per cell
vox = [[[None]*NZ for _ in range(NY)] for _ in range(NX)]
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            vox[i][j][k] = sphere(
                pos=cell_world(i, j, k), radius=0.24,
                color=color.red, opacity=0, emissive=True
            )

# ── Laplace-Domain PID  (Bilinear / Tustin Discretisation) ───────
#
#   Continuous:   U(s)   Kd·N·s
#                 ──── = Kp + Ki/s + ─────
#                 E(s)              s + N
#
#   Tustin:  s  →  (2/dt)·(z−1)/(z+1)
#
#   Integral term   I[n] = I[n−1] + (dt/2)·(e[n] + e[n−1])
#   Derivative term D[n] = α·D[n−1] + β·(e[n] − e[n−1])
#       α = (2 − N·dt)/(2 + N·dt)
#       β = 2·Kd·N  / (2 + N·dt)

Kp, Ki, Kd = 3.8, 0.05, 2.0
N_filt      = 20.0
I_CLAMP     = 60.0         # anti-windup bound

_I  = 0.0                  # integrator state
_D  = 0.0                  # derivative filter state
_pe = 0.0                  # previous error

def laplace_pid(error, dt):
    global _I, _D, _pe
    _I  = float(np.clip(_I + (dt/2.0)*(error + _pe), -I_CLAMP, I_CLAMP))
    a   = (2.0 - N_filt*dt) / (2.0 + N_filt*dt)
    b   = (2.0 * Kd * N_filt) / (2.0 + N_filt*dt)
    _D  = a*_D + b*(error - _pe)
    _pe = error
    return Kp*error + Ki*_I + _D

# ── Leak Management ───────────────────────────────────────────────
leaks     = []             # list of dicts
MAX_LEAKS = 3

def add_leak():
    """Add one random leak source (up to MAX_LEAKS active at once)."""
    if len(leaks) >= MAX_LEAKS:
        old = leaks.pop(0)
        old['marker'].visible = False
    gi = (random.randint(0, NX-1),
          random.randint(NY//2, NY-1),
          random.randint(0, NZ-1))
    lw     = cell_world(*gi)
    mask   = gaussian_mask(*gi, sigma=1.8)
    marker = sphere(pos=lw, radius=0.50, color=color.yellow,
                    emissive=True, opacity=0.88)
    leaks.append({
        'rate':   random.uniform(12.0, 22.0),
        'mask':   mask,
        'gi':     gi,
        'marker': marker
    })
    trigger_shockwave(lw)

# ── Shock-wave Ring System ────────────────────────────────────────
shockwaves   = []
SHOCK_LIFETIME = 1.25

def trigger_shockwave(pos):
    """Spawn two concentric expanding rings at pos."""
    for offset in [0.0, 0.18]:
        r = ring(
            pos=pos + vector(0, offset, 0),
            axis=vector(0, 1, 0),
            radius=0.35, thickness=0.22,
            color=color.yellow, emissive=True, opacity=0.92
        )
        shockwaves.append({
            'ring':     r,
            'age':      random.uniform(0, 0.08),
            'lifetime': SHOCK_LIFETIME + random.uniform(-0.15, 0.25)
        })

def update_shockwaves(dt):
    dead = []
    for sw in shockwaves:
        sw['age'] += dt
        t = sw['age'] / sw['lifetime']
        if t >= 1.0:
            sw['ring'].visible = False
            dead.append(sw)
        else:
            sw['ring'].radius    = 0.35 + t * 6.5
            sw['ring'].thickness = max(0.008, 0.22 * (1.0 - t))
            sw['ring'].opacity   = (1.0 - t**0.65) * 0.88
    for sw in dead:
        shockwaves.remove(sw)

# ── Flow-field Arrows (advection visualisation) ───────────────────
flow_arrows = []
SUCTION_PT  = P_BASE + vector(0, 1.6, 0)    # mid-purifier suction

for i in range(0, NX, 2):
    for k in range(0, NZ, 2):
        j  = NY // 2
        wp = cell_world(i, j, k)
        a  = arrow(
            pos=wp, axis=vector(0, 0, 0.001),
            color=color.cyan, shaftwidth=0.07,
            opacity=0.0, emissive=True
        )
        flow_arrows.append((a, i, j, k))

# ── Particle Smoke System ─────────────────────────────────────────
particles = []
MAX_PART  = 50

def spawn_particles():
    for lk in leaks:
        if lk['rate'] < 0.5 or len(particles) >= MAX_PART:
            continue
        lw = cell_world(*lk['gi'])
        p  = sphere(
            pos=lw + vector(random.uniform(-0.4, 0.4),
                            random.uniform(-0.25, 0.25),
                            random.uniform(-0.4, 0.4)),
            radius=0.18,
            color=color.yellow, opacity=0.78, emissive=True
        )
        p.vel  = vector(
            random.uniform(-0.5, 0.5),
            random.uniform(0.05, 0.75),
            random.uniform(-0.5, 0.5)
        )
        p.life = 1.0
        attach_trail(p, retain=18, radius=0.06, color=color.orange)
        particles.append(p)

# ── Status / Math Labels ─────────────────────────────────────────
status_lbl = label(
    pos=vector(0, 7.3, 0), text="✓  AIR QUALITY: 100% SAFE",
    color=color.green, box=False, height=16
)
math_lbl = label(
    pos=vector(-14, 5.8, 0), align='left',
    text="", color=color.cyan, box=False, height=10
)
intg_lbl = label(
    pos=vector(-14, 0.8, 0), align='left',
    text="", color=color.yellow, box=False, height=10
)
label(
    pos=vector(0, -6.9, 0),
    text="[ T ] Trigger Leak (max 3 active)   |   [ R ] Reset All",
    color=color.white, box=False, height=11
)

# ── Live PID Sliders (caption area) ──────────────────────────────
scene.append_to_caption('\n\n')

def set_kp(s):
    global Kp
    Kp = s.value
    _kp_txt.text = f'  {Kp:.2f}'

def set_ki(s):
    global Ki
    Ki = s.value
    _ki_txt.text = f'  {Ki:.3f}'

def set_kd(s):
    global Kd
    Kd = s.value
    _kd_txt.text = f'  {Kd:.2f}'

scene.append_to_caption('  Kp ')
slider(min=0.5,  max=8.0,  value=Kp, length=170, bind=set_kp)
_kp_txt = wtext(text=f'  {Kp:.2f}')

scene.append_to_caption('      Ki ')
slider(min=0.0,  max=0.5,  value=Ki, length=170, bind=set_ki)
_ki_txt = wtext(text=f'  {Ki:.3f}')

scene.append_to_caption('      Kd ')
slider(min=0.0,  max=5.0,  value=Kd, length=170, bind=set_kd)
_kd_txt = wtext(text=f'  {Kd:.2f}')

# ── Live Graphs ───────────────────────────────────────────────────
g = graph(
    title="T(t) = ∭C(x,y,z,t) dV   vs.   Laplace-PID  u(t)",
    width=1000, height=260,
    xtitle="Time  t  (s)", ytitle="Level", fast=True
)
curve_T  = gcurve(color=color.red,    label="T(t) — Total Toxin Mass  [triple integral]")
curve_u  = gcurve(color=color.cyan,   label="u(t) — Fan Speed %  [Laplace-Bilinear PID]")
curve_Cm = gcurve(color=color.orange, label="C_max × 8 — Peak Concentration in Field")

# ── Key Bindings ─────────────────────────────────────────────────
def on_key(evt):
    global C, _I, _D, _pe
    k = evt.key
    if k in ('t', 'T'):
        add_leak()
        status_lbl.text  = f"⚠  TOXIC LEAK DETECTED  ({len(leaks)} active)"
        status_lbl.color = color.red
    elif k in ('r', 'R'):
        C[:] = 0.0
        _I = _D = _pe = 0.0
        for lk in leaks:
            lk['marker'].visible = False
            lk['rate'] = 0.0
        leaks.clear()
        for p in particles:
            p.visible = False
        particles.clear()
        status_lbl.text  = "✓  ROOM RESET — AIR CLEARED"
        status_lbl.color = color.green

scene.bind('keydown', on_key)

# ── Simulation bookkeeping ────────────────────────────────────────
fan_speed = 0.0
sim_time  = 0.0
dt        = 0.03          # physics time-step [s]

combined_source = np.zeros((NX, NY, NZ))

# =================================================================
#  MAIN LOOP
# =================================================================
while True:
    rate(1 / dt)           # 33 physics updates per real second

    # ─ 1. Triple Integration  T(t) = ∭ C(x,y,z,t) dx dy dz ──────
    #      NumPy sum replaces 3 nested Python loops
    T_total = float(np.sum(C) * dV)

    # ─ 2. Laplace-Bilinear PID → fan control signal u(t) ─────────
    fan_speed = float(np.clip(laplace_pid(T_total, dt), 0.0, 100.0))

    # ─ 3. PDE Step  ∂C/∂t = D·∇²C + Σ src_k − sink  (NumPy) ─────
    combined_source[:] = 0.0
    for lk in leaks:
        combined_source += lk['rate'] * lk['mask']

    lap  = laplacian_np(C)
    sink = (fan_speed / 100.0) * sink_mask * C
    C    = np.maximum(0.0, C + (D_diff*lap + combined_source - sink) * dt)

    # Decay all leak rates naturally
    for lk in leaks:
        lk['rate'] = max(0.0, lk['rate'] - 0.18 * dt)
        if lk['rate'] < 0.08:
            lk['marker'].opacity = max(0.0, lk['marker'].opacity - dt * 0.8)

    C_max  = max(float(C.max()), 1e-9)
    danger = min(T_total / 220.0, 1.0)

    # ─ 4. Voxel Spheres: colour + size + opacity ──────────────────
    inv_max = 1.0 / C_max
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                cn = float(C[i, j, k]) * inv_max   # 0..1 normalised
                s  = vox[i][j][k]
                s.opacity = min(cn * 0.88, 0.88)
                s.radius  = 0.11 + cn * 0.44
                # Colour ramp: teal (safe) → yellow → red (danger)
                if cn < 0.5:
                    t2 = cn * 2.0
                    s.color = vector(t2, 1.0 - t2*0.22, 1.0 - t2)
                else:
                    t2 = (cn - 0.5) * 2.0
                    s.color = vector(1.0, 1.0 - t2*0.88, 0.0)

    # ─ 5. Fan Blades: spin speed proportional to u(t) ─────────────
    for fb in fan_blades:
        fb.rotate(
            angle=math.radians(fan_speed * 0.68),
            axis=vector(0, 1, 0),
            origin=BLADE_HUB
        )

    # ─ 6. Purifier Rings: phase-staggered pulse ───────────────────
    for idx, r in enumerate(pur_rings):
        phase = math.sin(sim_time * 5.5 + idx * 1.15)
        r.opacity = (0.35 + 0.40*phase) * (fan_speed / 100.0)

    pur_body.color   = vector(danger, 0.42*(1-danger), 1.0-danger)
    pur_dome.color   = vector(1-danger*0.65, 1-danger*0.45, 1.0)
    pur_glow.color   = vector(danger, 0.5, 1.0-danger)
    pur_glow.opacity = 0.04 + danger * 0.14

    # ─ 7. Room Danger Pulse ───────────────────────────────────────
    if danger > 0.42:
        pb = 0.5 + 0.5 * math.sin(sim_time * 11.5)
        room.color   = vector(1.0, 1.0 - danger*pb*0.95, 0.0)
        room.opacity = 0.07 + danger * pb * 0.30
    else:
        room.color   = vector(1.0, 1.0 - danger*0.72, 1.0 - danger*0.72)
        room.opacity = 0.05 + danger * 0.12

    # ─ 8. Shock-wave Rings ────────────────────────────────────────
    update_shockwaves(dt)

    # ─ 9. Flow-field Arrows ───────────────────────────────────────
    for (a, i, j, k) in flow_arrows:
        cn_local = float(C[i, j, k]) * inv_max
        toward   = SUCTION_PT - a.pos
        tm       = mag(toward)
        if tm > 0.01:
            strength = cn_local * (fan_speed / 100.0) * 0.95
            a.axis   = (toward / tm) * max(strength, 0.001)
        a.opacity = min(cn_local * 1.25, 0.80)
        # Arrow colour: cyan when calm, orange-red when heavy toxin load
        a.color   = vector(cn_local, 1.0 - cn_local*0.55, 1.0 - cn_local)

    # ─ 10. Particle Smoke ────────────────────────────────────────
    if sim_time % 0.18 < dt:
        spawn_particles()

    dead = []
    for p in particles:
        p.life -= dt * 0.30
        if p.life <= 0:
            p.visible = False
            dead.append(p)
        else:
            toward = SUCTION_PT - p.pos
            tm     = mag(toward)
            if tm > 0.01:
                p.vel += (toward / tm) * (fan_speed / 100.0) * 0.08
            p.pos    += p.vel * dt
            p.opacity = p.life * 0.78
    for p in dead:
        particles.remove(p)

    # ─ 11. Status & Math Labels ───────────────────────────────────
    n_leaks = len(leaks)
    if danger > 0.55:
        status_lbl.text  = (
            f"⚠  DANGER — CRITICAL   "
            f"T = {T_total:.2f}   Leaks: {n_leaks}"
        )
        status_lbl.color = color.red
    elif danger > 0.07:
        status_lbl.text  = (
            f"PURIFYING…   "
            f"T = {T_total:.2f}   Fan: {fan_speed:.0f}%   Leaks: {n_leaks}"
        )
        status_lbl.color = color.orange
    else:
        status_lbl.text  = "✓  AIR QUALITY: 100% SAFE"
        status_lbl.color = color.green

    math_lbl.text = (
        "Laplace PID  (Bilinear / Tustin Transform)\n"
        "  U(s)/E(s) = Kp + Ki/s + Kd·N·s/(s+N)\n"
        f"  Kp={Kp:.2f}   Ki={Ki:.3f}   Kd={Kd:.2f}   N={N_filt:.0f}\n"
        f"  s → (2/dt)·(z−1)/(z+1)   [dt={dt}]\n"
        f"  error  e(t)  = {T_total:.4f}\n"
        f"  output u(t)  = {fan_speed:.1f} %\n"
        f"  I-state      = {_I:.3f}   [±{I_CLAMP} clamp]\n"
        f"  D-state      = {_D:.3f}"
    )

    intg_lbl.text = (
        "Triple Integration  (NumPy)\n"
        "  T(t) = ∭_V C(x,y,z,t) dx dy dz\n"
        f"  Grid {NX}×{NY}×{NZ} = {NX*NY*NZ} cells   dV = {dV:.2f} m³\n"
        f"  T(t)  = {T_total:.5f}\n"
        f"  C_max = {C_max:.5f}\n\n"
        "3-D Diffusion PDE  (NumPy Laplacian)\n"
        "  ∂C/∂t = D·∇²C + Σ_k src_k − sink\n"
        f"  D = {D_diff}   Active leaks: {n_leaks}"
    )

    # ─ 12. Graph ─────────────────────────────────────────────────
    curve_T.plot( sim_time, T_total)
    curve_u.plot( sim_time, fan_speed)
    curve_Cm.plot(sim_time, C_max * 8.0)

    sim_time += dt
