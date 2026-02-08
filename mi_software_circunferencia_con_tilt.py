# -*- coding: utf-8 -*-
"""
Versión con TILT SUBPLOT: muestra solo la trayectoria óptima (r_opt) y dibuja una circunferencia cuyo radio
es el del primer punto óptimo. Incluye subgráfica de inclinación (tilt) de ambos dispositivos en la animación.

Elimina la trayectoria efectiva (r_eff) y el suavizado temporal.
"""

# Copia de mi software_circunfrencia.py (modificada para solo óptimo + circunferencia + tilt subplot)

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import csv as _csv
from pathlib import Path as _Path

# Default location (Madrid) if not specified
LAT_DEF = 40.4168
LON_DEF = -3.7038

Vec3 = Tuple[float, float, float]

@dataclass
class Config:
    tower_height: float
    r_min: float
    r_max: float
    lambda_dist: float
    v_max: float = 0.6
    w_max: float = 0.8
    dr_max: float = 0.6
    r_alpha: float = 0.3
    track_width: float = 0.5
    wheel_radius: float = 0.08
    gear_ratio: float = 1.0

@dataclass
class Command:
    t: datetime
    x: float
    y: float
    psi: float
    v: float
    w: float
    r: float
    n_az_deg: float
    tilt_deg: float

# small numeric constants
EPS_DEN = 1e-6
R_DOT_CLAMP = 1e6

# Helper functions (lightweight implementations)

def unit(v):
    if v is None:
        return None
    if len(v) == 3:
        x, y, z = v
        n = math.sqrt(x*x + y*y + z*z)
        if n == 0.0:
            return None
        return (x/n, y/n, z/n)
    else:
        x, y = v[0], v[1]
        n = math.hypot(x, y)
        if n == 0.0:
            return None
        return (x/n, y/n)

def azimuth_EN_from_vector(v):
    deg = math.degrees(math.atan2(v[0], v[1]))  # atan2(E,N) en ENU
    return (deg+360.0)%360.0

def az_from_xy_deg(x, y):
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def wrap_angle(a):
    while a <= -math.pi:
        a += 2*math.pi
    while a > math.pi:
        a -= 2*math.pi
    return a

def sunpos(y,m,d,hh,mm,ss,lat,lon)->Tuple[float,float]:
    rad = math.pi/180.0
    dH = hh + (mm + ss/60.0)/60.0
    a1 = (m-14)//12
    a2 = (1461*(y+4800+a1))//4 + (367*(m-2-12*a1))//12 - (3*((y+4900+a1)//100))//4 + d - 32075
    JD = float(a2) + dH/24.0 - 0.5
    EJD = JD - 2451545.0
    Om = 2.1429 - 0.0010394594*EJD
    L = 4.8950630 + 0.017202791698*EJD
    M = 6.2400600 + 0.0172019699*EJD
    lam = L + 0.03341607*math.sin(M) + 0.00034894*math.sin(2*M) - 0.0001134 - 0.0000203*math.sin(Om)
    eps = 0.4090928 - 6.214e-9*EJD + 0.0000396*math.cos(Om)
    y_ = math.cos(eps)*math.sin(lam); x_ = math.cos(lam)
    ra = math.atan2(y_, x_);  ra += 2*math.pi if ra<0 else 0
    dec = math.asin(math.sin(eps)*math.sin(lam))
    GMST = 6.6974243242 + 0.0657098283*EJD + dH
    LMST = (GMST*15.0 + lon)*rad
    H = LMST - ra
    phi = lat*rad
    z = math.acos(math.cos(phi)*math.cos(H)*math.cos(dec) + math.sin(dec)*math.sin(phi))
    y_ = -math.sin(H); x_ = math.tan(dec)*math.cos(phi) - math.sin(phi)*math.cos(H)
    A = math.atan2(y_, x_);  A += 2*math.pi if A<0 else 0
    par = (6371.01/149_597_890.0)*math.sin(z)
    return A, (z+par)

def sun_dir_ENU(dt: datetime, lat: float, lon: float)->Tuple[float,float,float,float]:
    A,z = sunpos(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second+dt.microsecond/1e6,lat,lon)
    sz = math.cos(z)
    sxy = math.sin(z)
    sx, sy, szU = math.sin(A)*sxy, math.cos(A)*sxy, sz
    return sx, sy, szU, A


def position_from_sun(A_sun: float, radius: float)->Vec3:
    beta = (A_sun + math.pi)%(2*math.pi)
    east  = radius*math.sin(beta)
    north = radius*math.cos(beta)
    return (east, north, 0.0)

def efficiency_for_radius(r: float, A_sun: float, s: Vec3, H: float, lam: float, r_max: float)->float:
    pos = position_from_sun(A_sun, r)
    tx,ty,tz = -pos[0], -pos[1], H
    tdir = unit((tx,ty,tz))
    if tdir is None: return 0.0
    n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2]))
    if n is None: return 0.0
    cos_eff = max(0.0, min(1.0, s[0]*n[0]+s[1]*n[1]+s[2]*n[2]))
    penalty = float(lam) * (r / r_max)**2
    return max(0.0, cos_eff - penalty)

def optimize_radius(A_sun: float, s: Vec3, cfg: Config)->float:
    a, b = cfg.r_min, cfg.r_max
    if b <= a: return a
    gr = (math.sqrt(5.0)-1.0)/2.0
    c = b - gr*(b-a); d = a + gr*(b-a)
    fc = efficiency_for_radius(c, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    fd = efficiency_for_radius(d, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    for _ in range(28):
        if fc < fd:
            a = c; c = d; fc = fd
            d = a + gr*(b-a); fd = efficiency_for_radius(d, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
        else:
            b = d; d = c; fd = fc
            c = b - gr*(b-a); fc = efficiency_for_radius(c, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    return (a+b)/2.0

# --- Planificador: solo r_opt (sin r_eff) ---
def plan_commands(start: datetime, end: datetime, step_s: float, lat: float, lon: float,
                  cfg: Config, mode: str="heading_follow"
                 ) -> Tuple[List[Command], List[Tuple[float,float]], List[Tuple[float,float]], List[datetime], List[float], List[float], List[float], List[float], List[bool], List[bool], List[float]]:
    dt = timedelta(seconds=step_s)
    t = start
    cmds: List[Command] = []

    opt_pts:  List[Tuple[float,float]] = []
    pred_pts: List[Tuple[float,float]] = []
    ts:       List[datetime] = []
    eta_opt:  List[float] = []
    eta_pred: List[float] = []
    eta_eff:  List[float] = []

    den_series: List[float] = []
    sat_series: List[bool]  = []
    sing_series: List[bool] = []
    rda_series: List[float] = []

    r_prev: Optional[float] = None
    psi_prev: Optional[float] = None
    A_prev: Optional[float] = None
    x_prev = y_prev = None

    while t <= end:
        sx,sy,sz,A = sun_dir_ENU(t, lat, lon)
        if sz <= 0.0:
            t += dt; continue
        s = (sx,sy,sz)

        r_opt = optimize_radius(A, s, cfg)
        pos_opt = position_from_sun(A, r_opt)
        eta_o = efficiency_for_radius(r_opt, A, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
        opt_pts.append((pos_opt[0], pos_opt[1]))

        # rumbo y heading clamp (igual que antes)
        r_base = r_opt if r_prev is None else r_prev
        pos_base = position_from_sun(A, r_base)
        tx,ty,tz = -pos_base[0], -pos_base[1], cfg.tower_height
        tdir = unit((tx,ty,tz))
        n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2])) if tdir else None
        if n is None:
            t += dt; continue
        n_az_rad = math.radians(azimuth_EN_from_vector(n))
        psi_d = wrap_angle(n_az_rad - math.pi/2)
        psi_for_r = psi_d if psi_prev is None else wrap_angle(0.9*psi_prev + 0.1*psi_d)

        if mode == "heading_follow" and A_prev is not None and r_prev is not None:
            beta = A + math.pi
            T = math.tan(psi_for_r)
            den = (T*math.sin(beta) - math.cos(beta))
            dA  = wrap_angle(A - A_prev)

            clamp_den = False
            clamp_sat = False

            if abs(den) < EPS_DEN:
                clamp_den = True
                r_dot_A_raw = 0.0
            else:
                r_dot_A_raw = - r_prev * ((T*math.cos(beta) + math.sin(beta)) / den)
                if abs(r_dot_A_raw) > R_DOT_CLAMP:
                    clamp_sat = True

            if clamp_den:
                r_dot_A = 0.0
            else:
                r_dot_A = max(-R_DOT_CLAMP, min(R_DOT_CLAMP, r_dot_A_raw))

            r_pred = r_prev + r_dot_A * dA

            den_series.append(abs(den))
            sat_series.append(clamp_sat)
            sing_series.append(clamp_den)
            rda_series.append(r_dot_A_raw)
        else:
            r_pred = r_opt
            den_series.append(1.0)
            sat_series.append(False)
            sing_series.append(False)
            rda_series.append(0.0)

        pos_pred = position_from_sun(A, r_pred)
        eta_p = efficiency_for_radius(r_pred, A, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
        pred_pts.append((pos_pred[0], pos_pred[1]))

        # --- Eliminamos suavizado temporal: usamos r_final = r_opt ---
        r_final = r_opt

        # Pose final y comandos (usando r_final)
        pos = position_from_sun(A, r_final)
        tx,ty,tz = -pos[0], -pos[1], cfg.tower_height
        tdir = unit((tx,ty,tz))
        n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2])) if tdir else None
        if n is None:
            t += dt; continue
        n_az_rad = math.radians(azimuth_EN_from_vector(n))
        psi_d = wrap_angle(n_az_rad - math.pi/2)

        if psi_prev is None or x_prev is None:
            v_cmd = 0.0; w_cmd = 0.0
        else:
            dpsi = wrap_angle(psi_d - psi_prev)
            w_cmd = max(-cfg.w_max, min(cfg.w_max, dpsi/dt.total_seconds()))
            dx = pos[0] - x_prev; dy = pos[1] - y_prev
            dist = max(1e-4, math.hypot(dx, dy))
            kappa_est = abs(dpsi) / max(1e-3, dist)
            v_curv = cfg.w_max / max(1e-3, kappa_est)
            v_cmd = min(cfg.v_max, v_curv)
            if abs(w_cmd) > 0.8*cfg.w_max:
                v_cmd *= 0.6

        cmds.append(Command(
            t=t, x=pos[0], y=pos[1], psi=psi_d,
            v=v_cmd, w=w_cmd, r=r_final,
            n_az_deg=math.degrees(n_az_rad), tilt_deg=math.degrees(math.acos(max(-1.0,min(1.0,n[2]))))
        ))

        ts.append(t)
        eta_opt.append(eta_o)
        eta_pred.append(eta_p)
        eta_eff.append(eta_o)  # igual que óptima (no hay eficaz separada ahora)

        r_prev = r_final
        psi_prev = psi_d
        A_prev = A
        x_prev, y_prev = pos[0], pos[1]
        t += dt

    # prints y estadísticas (igual que antes)
    def mean(v): return sum(v)/len(v) if v else 0.0
    d_p  = [o - p  for o,p  in zip(eta_opt, eta_pred)]
    d_e  = [o - e  for o,e  in zip(eta_opt, eta_eff)]
    print(f"[eta] Delta(opt - pred clamp): media={mean(d_p):.4f}  max={max(d_p) if d_p else 0:.4f}")
    print(f"[eta] Delta(opt - eff)      : media={mean(d_e):.4f}  max={max(d_e) if d_e else 0:.4f}")

    total = len(ts)
    sat_pct  = 100.0 * sum(1 for b in sat_series  if b) / max(1,total)
    sing_pct = 100.0 * sum(1 for b in sing_series if b) / max(1,total)
    print(f"[CLAMP] singularidad (den) activo: {sing_pct:.1f}%   |  saturación |dr/dA|: {sat_pct:.1f}%")
    if den_series:
        dn_min, dn_med, dn_max = min(den_series), sorted(den_series)[len(den_series)//2], max(den_series)
        rda_abs = [abs(v) for v in rda_series]
        print(f"[diag] |den| min/med/max = {dn_min:.4g}/{dn_med:.4g}/{dn_max:.4g}")
        print(f"[diag] |r_dot_A_raw| min/med/max = {min(rda_abs):.3g}/{sorted(rda_abs)[len(rda_abs)//2]:.3g}/{max(rda_abs):.3g}  (limite={R_DOT_CLAMP})")

    return (cmds, opt_pts, pred_pts, ts, eta_opt, eta_pred, eta_eff,
            den_series, sat_series, sing_series, rda_series)

# (resto de utilidades: daylight_bounds, _spans_from_flags, _normal_y_angulos_desde_pos)


def daylight_bounds(day: datetime, lat: float, lon: float)->Tuple[datetime, datetime]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    t = start
    step = timedelta(minutes=1)
    first = last = None
    while t < end:
        _,_,sz,_ = sun_dir_ENU(t, lat, lon)
        if sz > 0.0:
            if first is None: first = t
            last = t
        t += step
    if first is None:
        return start, start
    return first - timedelta(minutes=5), last + timedelta(minutes=5)


def _spans_from_flags(flags, ts):
    spans = []
    start = None
    n = len(flags)
    for k, flag in enumerate(flags):
        if flag and start is None:
            start = max(k-1, 0)
        end_of_series = (k == n-1)
        if (start is not None) and (not flag or end_of_series):
            end_idx = k if not flag else k
            spans.append((ts[start], ts[end_idx]))
            start = None
    return spans


def _normal_y_angulos_desde_pos(pos_xy: Tuple[float,float], s: Tuple[float,float,float], H: float):
    x, y = pos_xy
    tx, ty, tz = -x, -y, H
    tdir = unit((tx, ty, tz))
    if tdir is None:
        return None, None, None
    n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2]))
    if n is None:
        return None, None, None
    n_az_deg = azimuth_EN_from_vector(n)
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
    psi_deg  = (n_az_deg - 90.0)
    while psi_deg <= -180.0: psi_deg += 360.0
    while psi_deg >   180.0: psi_deg -= 360.0
    return n_az_deg, tilt_deg, psi_deg


def _minimal_arc_from_angles(thetas, samples:int=200):
    """Dado una lista de ángulos (radianes), devuelve una lista de ángulos
    que representa el arco mínimo que contiene todos los puntos (evita el mayor hueco).
    Si hay un único ángulo, devuelve un arco corto centrado en él."""
    if not thetas:
        return []
    # normalizar a [0, 2π)
    ang = [(t if t >= 0 else t + 2*math.pi) for t in thetas]
    ang.sort()
    n = len(ang)
    if n == 1:
        center = ang[0]
        start = center - math.pi/12
        end = center + math.pi/12
        return [start + (end-start)*i/samples for i in range(samples+1)]
    # encontrar el mayor hueco entre ángulos consecutive
    max_gap = -1.0
    idx = 0
    for i in range(n):
        j = (i+1) % n
        if j > i:
            gap = ang[j] - ang[i]
        else:
            gap = (ang[j] + 2*math.pi) - ang[i]
        if gap > max_gap:
            max_gap = gap; idx = i
    # el arco mínimo es el complemento del mayor hueco
    start = ang[(idx+1) % n]
    end = ang[idx]
    if end <= start:
        end += 2*math.pi
    return [start + (end-start)*i/samples for i in range(samples+1)]

# Main simplificado y graficado sólo óptimo + circunferencia

def main():
    import argparse
    p = argparse.ArgumentParser("Planificador (optima + circunferencia + TILT SUBPLOT)")
    p.add_argument("--lat", type=float, default=LAT_DEF)
    p.add_argument("--lon", type=float, default=LON_DEF)
    p.add_argument("--tower-h", type=float, default=3.0)
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("--r-max", type=float, default=15.0)
    p.add_argument("--lambda-dist", type=float, default=0.12)
    p.add_argument("--v-max", type=float, default=0.6, help="Velocidad lineal máxima (m/s)")
    p.add_argument("--w-max", type=float, default=0.8, help="Velocidad angular máxima (rad/s)")
    p.add_argument("--dr-max", type=float, default=0.6, help="Límite |dr/dt| (m/s)")
    p.add_argument("--r-alpha", type=float, default=0.3, help="Constante alfa de filtrado exponencial de r")
    p.add_argument("--step-s", type=float, default=0.5, help="Paso de planificación [s]")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--mode", type=str, default="heading_follow",
                   choices=["point_heading","heading_follow"])
    p.add_argument("--out-csv-base", type=str, default="plan")
    p.add_argument("--animate", action="store_true", help="Mostrar animación de vectores sobre la trayectoria óptima")
    p.add_argument("--save-anim", type=str, default=None, help="Ruta para guardar animación (mp4 o gif)")
    p.add_argument("--fps", type=int, default=12, help="FPS para animación guardada")
    p.add_argument("--debug-anim", action="store_true", help="Debug: imprime trazas de la animación en consola")
    a = p.parse_args()

    fecha_str = input("Fecha de trabajo (YYYY-MM-DD) [Enter = hoy]: ").strip()
    if fecha_str:
        try:
            day0 = datetime.fromisoformat(fecha_str)
            day0 = day0.replace(hour=12, minute=0, second=0, microsecond=0)
        except Exception as e:
            print(f"Fecha inválida, uso hoy. Detalle: {e}")
            day0 = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        day0 = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    ans = input("¿Usar día completo (amanecer→atardecer)? [s/n]: ").strip().lower()
    full_day = ans in ("s", "si", "sí")
    if full_day:
        start, end = daylight_bounds(day0, a.lat, a.lon)
    else:
        width_h = 8.0
        start = day0 - timedelta(hours=width_h/2)
        end   = day0 + timedelta(hours=width_h/2)

    # Preguntar en consola si desea animación (si no fue especificado por CLI)
    try:
        if not a.animate:
            ans_anim = input("¿Quieres animación? [s/n]: ").strip().lower()
            a.animate = ans_anim in ("s", "si", "sí")
        if not a.save_anim and a.animate:
            # Por defecto guardar como gif (compatible sin ffmpeg)
            fecha_str = day0.strftime('%Y%m%d')
            a.save_anim = f"animacion_{fecha_str}.gif"
            print(f"(Guardando animación automáticamente como: {a.save_anim})")
    except Exception:
        # En ejecuciones no interactivas, preservar flags CLI
        pass

    cfg = Config(tower_height=a.tower_h, r_min=a.r_min, r_max=a.r_max, lambda_dist=a.lambda_dist,
                 v_max=a.v_max, w_max=a.w_max, dr_max=a.dr_max, r_alpha=a.r_alpha,
                 track_width=0.5, wheel_radius=0.08, gear_ratio=1.0)

    (cmds, opt_pts, pred_pts, ts, eta_opt, eta_pred, eta_eff,
     den_series, sat_series, sing_series, rda_series) = plan_commands(start, end, a.step_s, a.lat, a.lon, cfg, mode=a.mode)

    base = _Path(a.out_csv_base)

    def _write_csv(path, rows):
        path = _Path(f"{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["time","x","y","r","az_deg","psi_deg","n_az_deg","tilt_deg"])
            w.writerows(rows)
        print(f"[CSV] {path}")

    rows_opt, rows_pred, rows_eff = [], [], []

    for k, t_k in enumerate(ts):
        sx, sy, sz, A = sun_dir_ENU(t_k, a.lat, a.lon)
        s_vec = (sx, sy, sz)

        x_opt, y_opt = opt_pts[k]
        r_opt_k = math.hypot(x_opt, y_opt)
        row = [_ for _ in []]
        row = [t_k.isoformat(timespec='minutes'), f"{x_opt:.6f}", f"{y_opt:.6f}", f"{r_opt_k:.6f}", f"{az_from_xy_deg(x_opt,y_opt):.6f}", "0.0", "0.0", "0.0"]
        rows_opt.append(row)

        x_p, y_p = pred_pts[k]
        r_p_k = math.hypot(x_p, y_p)
        row = [t_k.isoformat(timespec='minutes'), f"{x_p:.6f}", f"{y_p:.6f}", f"{r_p_k:.6f}", f"{az_from_xy_deg(x_p,y_p):.6f}", "0.0", "0.0", "0.0"]
        rows_pred.append(row)

        c = cmds[k]
        row = [t_k.isoformat(timespec='minutes'), f"{c.x:.6f}", f"{c.y:.6f}", f"{c.r:.6f}", f"{az_from_xy_deg(c.x,c.y):.6f}", f"{math.degrees(c.psi):.6f}", f"{c.n_az_deg:.6f}", f"{c.tilt_deg:.6f}"]
        rows_eff.append(row)


    _write_csv(f"{base}_opt.csv",  rows_opt)
    _write_csv(f"{base}_pred.csv", rows_pred)
    _write_csv(f"{base}_eff.csv",  rows_eff)

    if cmds:
        print(f"Intervalo planificado: {start.isoformat(timespec='minutes')} → {end.isoformat(timespec='minutes')}")

    # Gráfica: solo óptimo y circunferencia
    if not a.no_plot and cmds:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            xs_opt, ys_opt = [p[0] for p in opt_pts], [p[1] for p in opt_pts]
            fig, ax = plt.subplots(figsize=(18,10))
            ax.set_aspect('equal')
            # Expandir el área del eje para que ocupe más altura vertical dentro de la ventana
            ax.set_position([0.06, 0.04, 0.92, 0.92])
            ax.axhline(0, lw=0.8, ls='--', c='#aaa'); ax.axvline(0, lw=0.8, ls='--', c='#aaa')
            ax.scatter([0],[0], s=60, c='#d35400', zorder=3, label='Torre')
            ax.plot(xs_opt, ys_opt, lw=2.0, c='tab:blue', label='Óptimo η (r_opt)')
            if xs_opt and ys_opt:
                r0 = math.hypot(xs_opt[0], ys_opt[0])
                # Recortar la circunferencia al arco que cubren los puntos óptimos
                thetas = [math.atan2(x,y) for x, y in zip(xs_opt, ys_opt)]
                arc_thetas = _minimal_arc_from_angles(thetas, samples=360)
                xs_arc = [r0 * math.sin(t) for t in arc_thetas]
                ys_arc = [r0 * math.cos(t) for t in arc_thetas]
                ax.plot(xs_arc, ys_arc, lw=1.6, color='tab:green', alpha=0.9, label=f'Circunferencia (arc) r={r0:.2f} m')
                ax.plot([xs_opt[0]],[ys_opt[0]], 'o', color='tab:green', ms=6)
                # punto inicial en la circunferencia: colocar el CircDevice en el primer punto óptimo
                ax.plot([xs_opt[0]],[ys_opt[0]], marker='D', color='tab:orange', ms=6, label='CircDevice')
                # Ajustar límites del plot para ver más arriba/abajo sin necesidad de pan
                all_x = xs_opt + xs_arc
                all_y = ys_opt + ys_arc
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                rng_x = max_x - min_x; rng_y = max_y - min_y
                pad_base = max(1.0, 0.18 * max(rng_x, rng_y))
                pad_x = pad_base * 1.05
                pad_y = pad_base * 1.8
                ax.set_xlim(min_x - pad_x, max_x + pad_x)
                ax.set_ylim(min_y - pad_y, max_y + pad_y)
            ax.set_xlabel('Este (m)'); ax.set_ylabel('Norte (m)')
            ax.set_title('Planta: trayectoria óptima y circunferencia (radio primer punto óptimo)')
            ax.legend(loc='best')
            plt.tight_layout(); plt.show()

            # ------------------- Animación (opcional) -------------------
            def do_animation():
                # choose animation points depending on mode
                mode = getattr(a, 'animate_mode', 'opt') if hasattr(a, 'animate_mode') else 'opt'
                if mode == 'circ' and xs_opt and ys_opt:
                    r0 = math.hypot(xs_opt[0], ys_opt[0])
                    # arc built earlier
                    thetas = [math.atan2(x,y) for x, y in zip(xs_opt, ys_opt)]
                    arc_thetas = _minimal_arc_from_angles(thetas, samples=360)
                    anim_pts = [(r0*math.sin(t), r0*math.cos(t)) for t in arc_thetas]
                    anim_ts = [ts[min(len(ts)-1, int(i*(len(ts)/len(arc_thetas))))] for i in range(len(arc_thetas))]
                else:
                    # default: use optimal points
                    anim_pts = list(zip(xs_opt, ys_opt))
                    anim_ts = ts

                # automatic stride to keep frames reasonable
                n_frames = len(anim_pts)
                # Reduce frames más agresivamente para exportar
                desired_max = 800 if a.save_anim else 1500
                frame_stride = getattr(a, 'frame_stride', 1)
                if n_frames > desired_max and frame_stride == 1:
                    frame_stride = max(1, n_frames // desired_max)
                    print(f"(Aviso) demasiados frames ({n_frames}), se aplicará frame_stride={frame_stride} para acelerar")

                indices = list(range(0, n_frames, frame_stride))
                print(f"[Info] Animación con {len(indices)} frames (stride={frame_stride})")

                # figure and axes (taller)
                fig2, ax2 = plt.subplots(figsize=(18,10))
                ax2.set_aspect('equal')
                # Expandir el área del eje para que el plot sea visualmente más alto en la ventana
                # Make the plotting area taller within the window and draw guides
                ax2.set_position([0.06, 0.03, 0.88, 0.94])
                ax2.axhline(0, lw=0.6, ls='--', c='#ccc'); ax2.axvline(0, lw=0.6, ls='--', c='#ccc')
                # limits with separate padding for x and y (y padded more to show top/bottom)
                xs_anim = [p[0] for p in anim_pts]; ys_anim = [p[1] for p in anim_pts]
                all_x = xs_anim + xs_opt
                all_y = ys_anim + ys_opt
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                rng_x = max_x - min_x; rng_y = max_y - min_y
                pad_base = max(1.0, 0.18*max(rng_x, rng_y))
                pad_x = pad_base * 1.05
                pad_y = pad_base * 1.8
                ax2.set_xlim(min_x - pad_x, max_x + pad_x)
                ax2.set_ylim(min_y - pad_y, max_y + pad_y)
                ax2.set_xlabel('Este (m)'); ax2.set_ylabel('Norte (m)')
                ax2.set_title('Animacion: Sol, Normal y Rumbo (optimo + circunferencia)')

                # static: plot optimal path and tower and arc (always show both) 
                ax2.plot(xs_opt, ys_opt, lw=1.9, c='tab:blue', alpha=0.95, label='Optimo r_opt')
                xs_arc = ys_arc = []
                circle_pts = []
                circle_thetas = []
                circ_state = {'idx': None}  # use dict to allow mutation in nested function
                if xs_opt and ys_opt:
                    # compute arc for circle (same as before)
                    r0 = math.hypot(xs_opt[0], ys_opt[0])
                    thetas = [math.atan2(x,y) for x, y in zip(xs_opt, ys_opt)]
                    arc_thetas = _minimal_arc_from_angles(thetas, samples=360)
                    xs_arc = [r0 * math.sin(t) for t in arc_thetas]
                    ys_arc = [r0 * math.cos(t) for t in arc_thetas]
                    circle_pts = list(zip(xs_arc, ys_arc))
                    circle_thetas = [math.atan2(x, y) for x, y in circle_pts]
                    # alinear el inicio de la circunferencia con el primer punto óptimo
                    if circle_pts:
                        # encontrar índice del punto de la circunferencia más cercano al primer punto óptimo
                        def _sqdist(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2
                        start_pt = (xs_opt[0], ys_opt[0])
                        circ_state['idx'] = min(range(len(circle_pts)), key=lambda k: _sqdist(circle_pts[k], start_pt))
                    ax2.plot(xs_arc, ys_arc, lw=1.4, c='tab:green', alpha=0.95, label='Circunferencia (arc)')
                ax2.scatter([0],[0], s=60, c='#d35400', zorder=4, label='Torre')

                # active markers
                cur_pt, = ax2.plot([], [], 'o', color='tab:green', ms=8, label='Heliostato')
                cur_pt_b, = ax2.plot([], [], 'D', color='tab:orange', ms=7, label='CircDevice')
                # small quivers; lengths will be set explicitly (reduced sizes)
                q_sun = ax2.quiver([0],[0],[0],[0], color='gold', scale=1, scale_units='xy', width=0.004)
                q_norm = ax2.quiver([0],[0],[0],[0], color='limegreen', scale=1, scale_units='xy', width=0.005)
                q_psi  = ax2.quiver([0],[0],[0],[0], color='deepskyblue', scale=1, scale_units='xy', width=0.005)
                q_norm_b = ax2.quiver([0],[0],[0],[0], color='olive', scale=1, scale_units='xy', width=0.0045)
                q_psi_b  = ax2.quiver([0],[0],[0],[0], color='cadetblue', scale=1, scale_units='xy', width=0.0045)
                # sun lines (primary and secondary) and sun marker
                line_sun, = ax2.plot([], [], lw=0.9, color='gold', alpha=0.9, zorder=1)
                # linea visible desde el sol al dispositivo circunferencial (más ancha y encima)
                line_sun_b, = ax2.plot([], [], lw=1.2, color='goldenrod', alpha=0.95, zorder=3)
                # rayo reflejado desde el CircDevice hacia la torre
                line_ref_b, = ax2.plot([], [], lw=1.2, color='tab:red', alpha=0.9, zorder=2)
                sun_dot, = ax2.plot([], [], 'o', color='gold', ms=6)
                txta = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
                ax2.legend(loc='best')

                # ===== TILT SUBPLOT (nuevo) =====
                ax_tilt = fig2.add_axes([0.90, 0.18, 0.08, 0.62])
                ax_tilt.set_xticks([]); ax_tilt.set_yticks([])
                ax_tilt.set_xlim(-0.6, 0.6); ax_tilt.set_ylim(-0.6, 0.6)
                ax_tilt.set_title('Tilt (deg)', fontsize=9, color='gray')
                # static labels
                ax_tilt.text(0.05, 0.82, 'Helio', transform=ax_tilt.transAxes, va='center', fontsize=8, color='tab:green')
                ax_tilt.text(0.05, 0.32, 'CircD', transform=ax_tilt.transAxes, va='center', fontsize=8, color='tab:orange')
                # tilt lines (will be set in update)
                tilt_line_a, = ax_tilt.plot([], [], lw=2.5, color='tab:green', solid_capstyle='round')
                tilt_line_b, = ax_tilt.plot([], [], lw=2.5, color='tab:orange', solid_capstyle='round')
                tilt_txt_a = ax_tilt.text(0.50, 0.82, '', transform=ax_tilt.transAxes, va='center', fontsize=8, weight='bold')
                tilt_txt_b = ax_tilt.text(0.50, 0.32, '', transform=ax_tilt.transAxes, va='center', fontsize=8, weight='bold')

                # scaling: reduce sizes (smaller vectors)
                sun_scale = 0.20 * max(1.0, math.hypot(rng_x, rng_y))
                vec_scale = sun_scale * 0.30

                def update(frame_idx):
                    i = frame_idx
                    xk, yk = anim_pts[i]
                    t_k = anim_ts[i] if i < len(anim_ts) else ts[min(len(ts)-1, int(i*(len(ts)/len(anim_ts))))]
                    sx, sy, sz, A = sun_dir_ENU(t_k, a.lat, a.lon)
                    # prepare defaults
                    nx = ny = 0.0
                    psi_vx = psi_vy = 0.0
                    # helper to draw a small centered line at angle ang_deg (degrees)
                    def _rot_line(cx, cy, ang_deg, half_len=0.28):
                        a_r = math.radians(ang_deg)
                        dx = half_len * math.cos(a_r)
                        dy = half_len * math.sin(a_r)
                        return [cx - dx, cx + dx], [cy - dy, cy + dy]
                    tilt = float('nan')
                    # normal computed from pos
                    tx, ty, tz = -xk, -yk, cfg.tower_height
                    tdir = unit((tx, ty, tz))
                    if tdir is not None:
                        n = unit((sx + tdir[0], sy + tdir[1], sz + tdir[2]))
                        if n is not None:
                            nx, ny = n[0], n[1]
                            tilt = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
                            n_az_rad = math.radians(azimuth_EN_from_vector(n))
                            psi = wrap_angle(n_az_rad - math.pi/2)
                            psi_vx, psi_vy = math.sin(psi), math.cos(psi)
                    # use sunpos to place sun relative to tower (correct side of tower)
                    A_sun, z_sun = sunpos(t_k.year, t_k.month, t_k.day, t_k.hour, t_k.minute, t_k.second + t_k.microsecond/1e6, a.lat, a.lon)
                    sun_world_x = sun_scale * math.sin(A_sun)
                    sun_world_y = sun_scale * math.cos(A_sun)
                    line_sun.set_data([xk, sun_world_x], [yk, sun_world_y])
                    sun_dot.set_data([sun_world_x], [sun_world_y])
                    # set data for primary device
                    cur_pt.set_data([xk], [yk])
                    dx, dy = sun_world_x - xk, sun_world_y - yk
                    norm_sd = math.hypot(dx, dy) or 1.0
                    q_sun.set_offsets([[xk, yk]]); q_sun.set_UVC([dx / norm_sd * vec_scale * 1.1], [dy / norm_sd * vec_scale * 1.1])
                    q_norm.set_offsets([[xk, yk]]); q_norm.set_UVC([nx * vec_scale], [ny * vec_scale])
                    q_psi.set_offsets([[xk, yk]]); q_psi.set_UVC([psi_vx * vec_scale], [psi_vy * vec_scale])
                    # PRIMARY TILT INDICATOR
                    try:
                        xs_ta, ys_ta = _rot_line(0.0, 0.25, tilt)
                        tilt_line_a.set_data(xs_ta, ys_ta)
                        tilt_txt_a.set_text(f"{tilt:.1f}")
                    except Exception:
                        tilt_line_a.set_data([], []); tilt_txt_a.set_text('')
                    # secondary device on the circunference (phase-shifted)
                    if circle_pts:
                        # target: point on circle aligned with sun (opposite side from sun relative to tower)
                        # sun is at azimuth A, so device should be at A + pi (opposite side)
                        theta_des = (A + math.pi) % (2*math.pi)
                        def _ang_diff(a, b):
                            d = a - b
                            while d > math.pi: d -= 2*math.pi
                            while d < -math.pi: d += 2*math.pi
                            return abs(d)
                        tgt_idx = min(range(len(circle_pts)), key=lambda k: _ang_diff(circle_thetas[k], theta_des))
                        # advance circ_state['idx'] toward target with variable step (speed adapts to alignment needs)
                        if circ_state['idx'] is None:
                            # shouldn't happen, but fallback
                            circ_state['idx'] = 0
                        n_c = len(circle_pts)
                        diff = (tgt_idx - circ_state['idx']) % n_c
                        if diff > n_c/2:
                            diff -= n_c
                        max_step = 2
                        step = max(-max_step, min(max_step, int(round(diff * 0.3))))
                        circ_state['idx'] = (circ_state['idx'] + step) % n_c
                        xk_b, yk_b = circle_pts[circ_state['idx']]
                        cur_pt_b.set_data([xk_b], [yk_b])
                        # always draw a visible line between sun and this circ device
                        try:
                            line_sun_b.set_data([xk_b, sun_world_x], [yk_b, sun_world_y])
                            line_sun_b.set_zorder(3)
                        except Exception:
                            pass
                        # reflected ray from CircDevice to tower (0,0)
                        try:
                            line_ref_b.set_data([xk_b, 0.0], [yk_b, 0.0])
                        except Exception:
                            pass
                        # normal for b
                        txb, tyb, tzb = -xk_b, -yk_b, cfg.tower_height
                        tdir_b = unit((txb, tyb, tzb))
                        if tdir_b is not None:
                            nb = unit((sx + tdir_b[0], sy + tdir_b[1], sz + tdir_b[2]))
                            if nb is not None:
                                nx_b, ny_b = nb[0], nb[1]
                                q_norm_b.set_offsets([[xk_b, yk_b]]); q_norm_b.set_UVC([nx_b * vec_scale], [ny_b * vec_scale])
                                n_az_rad_b = math.radians(azimuth_EN_from_vector(nb))
                                psi_b = wrap_angle(n_az_rad_b - math.pi/2)
                                try:
                                    q_psi_b.set_offsets([[xk_b, yk_b]])
                                    q_psi_b.set_UVC([math.sin(psi_b) * vec_scale], [math.cos(psi_b) * vec_scale])
                                except Exception as ex:
                                    print(f"(Aviso) error actualizando q_psi_b en frame idx {i}: {ex}")
                                # SECONDARY TILT INDICATOR
                                try:
                                    tilt_b = math.degrees(math.acos(max(-1.0, min(1.0, nb[2]))))
                                    xs_tb, ys_tb = _rot_line(0.0, -0.25, tilt_b)
                                    tilt_line_b.set_data(xs_tb, ys_tb)
                                    tilt_txt_b.set_text(f"{tilt_b:.1f}")
                                except Exception:
                                    tilt_line_b.set_data([], []); tilt_txt_b.set_text('')
                    # update time/tilt text
                    txta.set_text(f"t={t_k.isoformat(timespec='minutes')}\nr={math.hypot(xk,yk):.2f} m  tilt={tilt:.1f} deg")
                    return cur_pt, cur_pt_b, q_sun, q_norm, q_psi, q_norm_b, q_psi_b, line_sun, line_sun_b, line_ref_b, sun_dot, txta, tilt_line_a, tilt_line_b, tilt_txt_a, tilt_txt_b

                def init_anim():
                    cur_pt.set_data([], [])
                    cur_pt_b.set_data([], [])
                    line_sun.set_data([], []); line_sun_b.set_data([], []); line_ref_b.set_data([], []); sun_dot.set_data([], [])
                    q_sun.set_offsets([[0,0]]); q_sun.set_UVC([0], [0])
                    q_norm.set_offsets([[0,0]]); q_norm.set_UVC([0], [0])
                    q_psi.set_offsets([[0,0]]); q_psi.set_UVC([0], [0])
                    q_norm_b.set_offsets([[0,0]]); q_norm_b.set_UVC([0], [0])
                    q_psi_b.set_offsets([[0,0]]); q_psi_b.set_UVC([0], [0])
                    tilt_line_a.set_data([], [])
                    tilt_line_b.set_data([], [])
                    tilt_txt_a.set_text('')
                    tilt_txt_b.set_text('')
                    txta.set_text('')
                    return cur_pt, cur_pt_b, q_sun, q_norm, q_psi, q_norm_b, q_psi_b, line_sun, line_sun_b, line_ref_b, sun_dot, txta, tilt_line_a, tilt_line_b, tilt_txt_a, tilt_txt_b

                # compute interval and frames according to speed / stride
                anim_speed = getattr(a, 'anim_speed', 1.0)
                base_interval = max(10, int(1000.0 / max(1, a.fps)))
                interval = max(5, int(base_interval / float(anim_speed)))

                # Use the list of indices directly as frames (comportamiento previo)
                frames = indices
                if not frames:
                    print('(Aviso) No hay frames para animar')
                    return
                print(f'(Info) iniciando animacion con {len(frames)} frames')
                anim = FuncAnimation(fig2, update, frames=frames, init_func=init_anim,
                                     interval=interval, blit=False, repeat=True)
                # keep reference to prevent GC
                fig2._anim_ref = anim
                try:
                    anim.event_source.start()
                except Exception:
                    pass

                # save or show
                if a.save_anim:
                    out = _Path(a.save_anim)
                    print(f"Guardando animacion → {out} (puede tardar unos segundos...)")
                    ext = out.suffix.lower()
                    try:
                        if ext in ('.mp4', '.mov'):
                            anim.save(str(out), dpi=100, writer='ffmpeg', fps=a.fps, bitrate=1800)
                        elif ext == '.gif':
                            try:
                                from matplotlib.animation import PillowWriter
                                anim.save(str(out), writer=PillowWriter(fps=a.fps), dpi=80)
                            except Exception:
                                anim.save(str(out), dpi=80, writer='imagemagick')
                        else:
                            anim.save(str(out), dpi=100)
                        print(f'✓ Guardado en: {out.absolute()}')
                    except Exception as ex:
                        print(f"(Aviso) error guardando animacion: {ex}")
                elif a.animate:
                    plt.show()

            # Ejecutar animacion si se pidio o se quiere guardar
            if a.animate or a.save_anim:
                do_animation()
        except Exception as ex:
            print(f"(Aviso) No se pudo dibujar: {ex}")

if __name__ == '__main__':
    main()
