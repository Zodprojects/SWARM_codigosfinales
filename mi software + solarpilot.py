# -*- coding: utf-8 -*-
"""
Script INTEGRADO: Comparación Modelo Propio vs SolarPILOT
=========================================================
Este script:
1. Calcula trayectorias (óptima y circular) con el modelo propio
2. Precalcula eficiencias de SolarPILOT para ambas trayectorias
3. Calcula eficiencias del modelo propio para los mismos puntos
4. Exporta CSVs comparativos
5. Genera gráficas de comparación

Uso:
    python integrated_comparison.py [opciones]

Opciones principales:
    --no-solarpilot      Desactivar SolarPILOT (solo modelo propio)
    --sp-samples N       Muestras por hora para SolarPILOT (default: 10)
    --step-s S          Paso de planificación en segundos (default: 0.5)
    
Ejemplo:
    python integrated_comparison.py --sp-samples 20
"""

import sys
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import csv

# Importar GUI de configuración
try:
    from config_gui import run_config_gui
    print("[OK] GUI de configuración importada correctamente")
except ImportError as e:
    print(f"[ERROR] No se pudo importar la GUI: {e}")
    print("Asegúrate de que 'config_gui.py' está en el mismo directorio")
    sys.exit(1)

# Importar funciones del modelo propio
try:
    from mi_software_circunferencia_con_tilt_eff import (
        Config, plan_commands, daylight_bounds, sun_dir_ENU,
        compute_efficiency_components, unit, LAT_DEF, LON_DEF
    )
    print("[OK] Modelo propio importado correctamente")
except ImportError as e:
    print(f"[ERROR] No se pudo importar el modelo propio: {e}")
    print("Asegúrate de que 'mi_software_circunferencia_con_tilt_eff.py' está en el mismo directorio")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN (Nota: ahora se obtiene desde la GUI)
# ============================================================================

# ============================================================================
# FUNCIONES DE ENJAMBRE
# ============================================================================

def get_hexagonal_swarm_offsets(separation):
    """
    Calcula los offsets para 7 heliostat en formación hexagonal.
    
    Configuración:
          6
       5  0  1
          2
       4     3
    
    Args:
        separation: Distancia centro a centro entre heliostat [m]
    
    Returns:
        Lista de 7 tuplas (offset_x, offset_y) para cada heliostat
    """
    # Heliostat central
    offsets = [(0.0, 0.0)]
    
    # 6 heliostat alrededor en patrón hexagonal
    angle_step = 60.0  # grados
    for i in range(6):
        angle_rad = math.radians(i * angle_step)
        offset_x = separation * math.cos(angle_rad)
        offset_y = separation * math.sin(angle_rad)
        offsets.append((offset_x, offset_y))
    
    return offsets


def apply_swarm_to_trajectory(x_traj, y_traj, swarm_offsets):
    """
    Aplica offsets del enjambre a una trayectoria base.
    
    Args:
        x_traj: Array de coordenadas X de la trayectoria base
        y_traj: Array de coordenadas Y de la trayectoria base
        swarm_offsets: Lista de tuplas (offset_x, offset_y) para cada heliostat
    
    Returns:
        swarm_trajectories: Lista de tuplas (x_array, y_array) para cada heliostat
    """
    swarm_trajectories = []
    
    for offset_x, offset_y in swarm_offsets:
        x_swarm = x_traj + offset_x
        y_swarm = y_traj + offset_y
        swarm_trajectories.append((x_swarm, y_swarm))
    
    return swarm_trajectories


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def nearest_traj_point(t_query, x_traj, y_traj, t_traj, tolerance=0.5):
    """
    Encuentra el punto más cercano en trayectoria para un tiempo dado.
    Devuelve None, None si el tiempo está fuera del rango válido.
    
    Args:
        tolerance: margen horario en horas (default 0.5h = 30 min)
    """
    mask = ~np.isnan(t_traj)
    if not mask.any():
        return None, None
    
    t_valid = t_traj[mask]
    t_min, t_max = t_valid.min(), t_valid.max()
    
    # Si está fuera del rango con tolerancia, devolver None
    if t_query < (t_min - tolerance) or t_query > (t_max + tolerance):
        return None, None
    
    idx = int(np.argmin(np.abs(t_valid - t_query)))
    idx_real = np.arange(len(t_traj))[mask][idx]
    return float(x_traj[idx_real]), float(y_traj[idx_real])


def precompute_solarpilot_for_traj(cp, r, t_list, x_traj, y_traj, t_traj, 
                                    XY, ids_list, month, day):
    """
    Precalcula eficiencias de SolarPILOT para una trayectoria completa.
    Incluye cálculo del heliostat fijo con mejor eficiencia promedio.
    """
    print(f"[SolarPILOT] Precalculando {len(t_list)} puntos...")
    
    result = {
        "eta_no_sb": [],
        "sol_id": [],
        "t_list": t_list
    }
    
    # Acumular eficiencias por heliostat para encontrar el mejor fijo
    eta_by_id = {}
    for hid in ids_list:
        eta_by_id[hid] = []
    
    # Caché de resultados detallados por timestep
    results_cache = {}
    
    n_skipped = 0
    
    for i, t_sub in enumerate(t_list):
        if i % max(1, len(t_list) // 10) == 0:
            print(f"  Progreso: {i}/{len(t_list)} ({100*i/len(t_list):.0f}%)")
        
        # Configurar tiempo y simular
        cp.data_set_number(r, "fluxsim.0.flux_month", float(month))
        cp.data_set_number(r, "fluxsim.0.flux_day", float(day))
        cp.data_set_number(r, "fluxsim.0.flux_hour", float(t_sub))
        cp.simulate(r)
        
        # Obtener resultados detallados
        mat, header = cp.detail_results(r, restype="matrix")
        H = [h.strip() for h in header]
        rows = [{H[j]: row[j] for j in range(len(H))} for row in mat]
        
        # Guardar en caché para extracción posterior del heliostat fijo
        results_cache[t_sub] = rows
        
        # Encontrar punto en trayectoria para este tiempo
        px, py = nearest_traj_point(t_sub, x_traj, y_traj, t_traj)
        
        # Si no hay punto válido, saltar
        if px is None or py is None:
            result["eta_no_sb"].append(np.nan)
            result["sol_id"].append(-1)
            n_skipped += 1
            continue
        
        # Encontrar heliostato más cercano
        d2 = (XY[:,0] - px)**2 + (XY[:,1] - py)**2
        closest_idx = int(np.argmin(d2))
        hel_id = ids_list[closest_idx]
        
        # Buscar fila correspondiente
        row_data = None
        for rw in rows:
            if str(rw.get("id", "")) == str(hel_id) or str(rw.get("heliostat_id", "")) == str(hel_id):
                row_data = rw
                break
        
        if row_data is None:
            result["eta_no_sb"].append(np.nan)
            result["sol_id"].append(hel_id)
            continue
        
        # Extraer eficiencias y factores individuales
        def _n01(v, d=1.0):
            try:
                x = float(v)
            except:
                return d
            if 1.0 < x <= 100.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        
        eta_all = _n01(row_data.get("efficiency", 0.0), d=0.0)
        sh_fac = _n01(row_data.get("shading", 1.0), d=1.0)
        bl_fac = _n01(row_data.get("blocking", 1.0), d=1.0)
        
        # Obtener factores individuales para reconstruir eficiencia sin sombras/bloqueos
        cos_fac = _n01(row_data.get("cosine", 1.0), d=1.0)
        int_fac = _n01(row_data.get("intercept", 1.0), d=1.0)
        refl_fac = _n01(row_data.get("reflectance", 1.0), d=1.0)
        atm_fac = _n01(row_data.get("attenuation", 1.0), d=1.0)
        cloud_fac = _n01(row_data.get("clouds", 1.0), d=1.0)
        
        # Reconstruir eficiencia sin sombras ni bloqueos directamente desde componentes
        # En lugar de dividir (que falla con sh=0), multiplicamos los factores base
        eta_no_sb = cos_fac * int_fac * refl_fac * atm_fac * cloud_fac
        
        # Clamp a [0,1] por seguridad
        eta_no_sb = max(0.0, min(1.0, eta_no_sb)) if np.isfinite(eta_no_sb) else 0.0
        
        result["eta_no_sb"].append(eta_no_sb)
        result["sol_id"].append(hel_id)
        
        # Acumular eficiencias de TODOS los heliostatos para encontrar el mejor fijo
        for rw in rows:
            try:
                hid_all = rw.get("id", rw.get("heliostat_id"))
                if hid_all in eta_by_id:
                    eta_all_hel = _n01(rw.get("efficiency", 0.0), d=0.0)
                    sh_fac_hel = _n01(rw.get("shading", 1.0), d=1.0)
                    bl_fac_hel = _n01(rw.get("blocking", 1.0), d=1.0)
                    cos_fac_hel = _n01(rw.get("cosine", 1.0), d=1.0)
                    int_fac_hel = _n01(rw.get("intercept", 1.0), d=1.0)
                    refl_fac_hel = _n01(rw.get("reflectance", 1.0), d=1.0)
                    atm_fac_hel = _n01(rw.get("attenuation", 1.0), d=1.0)
                    cloud_fac_hel = _n01(rw.get("clouds", 1.0), d=1.0)
                    eta_no_sb_hel = cos_fac_hel * int_fac_hel * refl_fac_hel * atm_fac_hel * cloud_fac_hel
                    eta_no_sb_hel = max(0.0, min(1.0, eta_no_sb_hel)) if np.isfinite(eta_no_sb_hel) else 0.0
                    eta_by_id[hid_all].append(eta_no_sb_hel)
            except:
                continue
    
    print(f"[SolarPILOT] Precálculo completado.")
    nan_count = sum(1 for x in result["eta_no_sb"] if not np.isfinite(x))
    if nan_count > 0:
        print(f"[WARN] {nan_count}/{len(result['eta_no_sb'])} puntos con NaN (problemas de división o factores inválidos)")
    
    # Determinar el conjunto de heliostatos asociados a la trayectoria
    # (aquellos que fueron elegidos como el más cercano en algún timestep)
    traj_ids = set([sid for sid in result["sol_id"] if sid is not None and sid != -1])
    if not traj_ids:
        # Si no se detectaron heliostatos cercanos, considerar todos los ids
        traj_ids = set(eta_by_id.keys())

    # Encontrar heliostat fijo con mejor eficiencia promedio entre los de la trayectoria
    best_fixed_id = None
    best_fixed_eta = -1.0
    for hid in traj_ids:
        etas = eta_by_id.get(hid, [])
        if not etas:
            continue
        valid_etas = [e for e in etas if np.isfinite(e)]
        if not valid_etas:
            continue
        mean_eta = float(np.mean(valid_etas))
        if mean_eta > best_fixed_eta:
            best_fixed_eta = mean_eta
            best_fixed_id = hid

    # Guardar ids de trayectoria en el resultado para trazabilidad
    result['traj_ids'] = list(traj_ids)
    
    # Extraer serie temporal del heliostat fijo
    fixed_eta_series = []
    if best_fixed_id is not None:
        print(f"[SolarPILOT] Heliostat fijo mejor: id={best_fixed_id}, η̄={best_fixed_eta:.4f}")
        
        def _n01(v, d=1.0):
            try:
                x = float(v)
            except:
                return d
            if 1.0 < x <= 100.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        
        for t_sub in t_list:
            if t_sub in results_cache:
                rows = results_cache[t_sub]
                # Buscar fila del heliostat fijo
                fixed_row = None
                for rw in rows:
                    if str(rw.get("id", "")) == str(best_fixed_id) or str(rw.get("heliostat_id", "")) == str(best_fixed_id):
                        fixed_row = rw
                        break
                
                if fixed_row is not None:
                    cos_fac = _n01(fixed_row.get("cosine", 1.0), d=1.0)
                    int_fac = _n01(fixed_row.get("intercept", 1.0), d=1.0)
                    refl_fac = _n01(fixed_row.get("reflectance", 1.0), d=1.0)
                    atm_fac = _n01(fixed_row.get("attenuation", 1.0), d=1.0)
                    cloud_fac = _n01(fixed_row.get("clouds", 1.0), d=1.0)
                    eta_no_sb_fixed = cos_fac * int_fac * refl_fac * atm_fac * cloud_fac
                    eta_no_sb_fixed = max(0.0, min(1.0, eta_no_sb_fixed)) if np.isfinite(eta_no_sb_fixed) else np.nan
                    fixed_eta_series.append(eta_no_sb_fixed)
                else:
                    fixed_eta_series.append(np.nan)
            else:
                fixed_eta_series.append(np.nan)
    else:
        print(f"[WARN] No se pudo determinar heliostat fijo")
        fixed_eta_series = [np.nan] * len(t_list)
    
    result["eta_fixed"] = fixed_eta_series
    result["fixed_id"] = best_fixed_id
    result["fixed_mean_eta"] = best_fixed_eta
    
    return result


def compute_my_model_series(t_list, x_traj, y_traj, t_traj, cfg, lat, lon, day):
    """
    Calcula serie de eficiencias con el modelo propio para un único heliostat.
    """
    print(f"[Modelo Propio] Calculando {len(t_list)} puntos...")
    
    result = {
        "eta_cos": [],
        "eta_att": [],
        "eta_int": [],
        "eta_total": [],
        "dist": [],
        "t_list": t_list
    }
    
    n_skipped = 0
    rT = (0.0, 0.0, cfg.tower_height)
    
    for i, t_sub in enumerate(t_list):
        if i % max(1, len(t_list) // 10) == 0:
            print(f"  Progreso: {i}/{len(t_list)} ({100*i/len(t_list):.0f}%)")
        
        # Convertir hora decimal a datetime
        hh = int(t_sub)
        mm = int((t_sub - hh) * 60)
        ss = int(((t_sub - hh) * 60 - mm) * 60)
        t_dt = day.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        
        # Vector solar
        sx, sy, sz, A = sun_dir_ENU(t_dt, lat, lon)
        if sz <= 0:
            result["eta_cos"].append(0.0)
            result["eta_att"].append(0.0)
            result["eta_int"].append(0.0)
            result["eta_total"].append(0.0)
            result["dist"].append(np.nan)
            continue
        
        # Posición en trayectoria
        px, py = nearest_traj_point(t_sub, x_traj, y_traj, t_traj)
        
        # Si no hay punto válido, agregar ceros
        if px is None or py is None:
            result["eta_cos"].append(0.0)
            result["eta_att"].append(0.0)
            result["eta_int"].append(0.0)
            result["eta_total"].append(0.0)
            result["dist"].append(np.nan)
            n_skipped += 1
            continue
        
        pA = (px, py, 0.0)
        
        # Calcular normal
        tx, ty, tz = -px, -py, cfg.tower_height
        tdir = unit((tx, ty, tz))
        if tdir is None:
            result["eta_cos"].append(0.0)
            result["eta_att"].append(0.0)
            result["eta_int"].append(0.0)
            result["eta_total"].append(0.0)
            result["dist"].append(np.nan)
            continue
        
        n = unit((sx + tdir[0], sy + tdir[1], sz + tdir[2]))
        if n is None:
            result["eta_cos"].append(0.0)
            result["eta_att"].append(0.0)
            result["eta_int"].append(0.0)
            result["eta_total"].append(0.0)
            result["dist"].append(np.nan)
            continue
        
        # Calcular eficiencia
        eff = compute_efficiency_components((sx, sy, sz), n, pA, rT, cfg)
        
        result["eta_cos"].append(eff["eta_cos"])
        result["eta_att"].append(eff["eta_att"])
        result["eta_int"].append(eff["eta_int"])
        result["eta_total"].append(eff["eta_total"])
        result["dist"].append(eff["dist"])
    
    print(f"[Modelo Propio] Cálculo completado.")
    if n_skipped > 0:
        print(f"[WARN] {n_skipped} puntos fuera de rango de trayectoria (saltados)")
    return result


def compute_swarm_model_series(t_list, swarm_trajectories, t_traj, cfg, lat, lon, day):
    """
    Calcula eficiencias para todos los heliostat del enjambre.
    
    Returns:
        Lista de resultados, uno por cada heliostat
    """
    print(f"\n[Modelo Propio - Enjambre] Calculando para {len(swarm_trajectories)} heliostat...")
    
    swarm_results = []
    for hel_idx, (x_traj, y_traj) in enumerate(swarm_trajectories):
        print(f"\n  Heliostat {hel_idx + 1}/{len(swarm_trajectories)}")
        result = compute_my_model_series(t_list, x_traj, y_traj, t_traj, cfg, lat, lon, day)
        swarm_results.append(result)
    
    print(f"\n[OK] Cálculo de enjambre completado")
    return swarm_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ========================================================================
    # PASO 0: Obtener configuración desde GUI
    # ========================================================================
    print("\n" + "="*70)
    print("LANZANDO INTERFAZ DE CONFIGURACIÓN")
    print("="*70)
    
    config = run_config_gui()
    
    if config is None:
        print("\n[INFO] Configuración cancelada por el usuario")
        return
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN RECIBIDA - INICIANDO SIMULACIÓN")
    print("="*70)
    
    # Extraer parámetros de la configuración
    use_sp = config['use_solarpilot']
    
    # ========================================================================
    # PASO 1: Preparar fecha y rango temporal desde configuración
    # ========================================================================
    print("\n" + "="*70)
    print("PASO 1: Configuración temporal")
    print("="*70)
    
    try:
        day0 = datetime.strptime(config['date'], '%Y-%m-%d')
        day0 = day0.replace(hour=12, minute=0, second=0, microsecond=0)
    except Exception as e:
        print(f"[WARN] Fecha inválida en configuración, usando hoy. Detalle: {e}")
        day0 = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    
    print(f"Fecha: {day0.strftime('%Y-%m-%d')}")
    print(f"Latitud: {config['lat']:.4f}°, Longitud: {config['lon']:.4f}°")
    print(f"SolarPILOT: {'ACTIVADO' if use_sp else 'DESACTIVADO'}")
    if use_sp:
        print(f"Muestras/hora SP: {config['sp_samples']}")
        print(f"Archivo meteorológico: {config['weather_file']}")
    
    full_day = config['full_day']
    if full_day:
        start, end = daylight_bounds(day0, config['lat'], config['lon'])
        print(f"Modo: Día completo (amanecer→atardecer)")
    else:
        width_h = 8.0
        start = day0 - timedelta(hours=width_h/2)
        end = day0 + timedelta(hours=width_h/2)
        print(f"Modo: Ventana de {width_h} horas")
    
    print(f"Intervalo: {start.strftime('%H:%M')} → {end.strftime('%H:%M')}")
    
    # ========================================================================
    # PASO 2: Calcular trayectorias con modelo propio
    # ========================================================================
    print("\n" + "="*70)
    print("PASO 2: Calculando trayectorias con modelo propio")
    print("="*70)
    
    # Crear configuración del modelo propio con parámetros de la GUI
    cfg = Config(
        tower_height=config['tower_height'],
        r_min=config['r_min'],
        r_max=config['r_max'],
        lambda_dist=config['lambda_dist'],
        v_max=config['v_max'],
        w_max=config['w_max'],
        receiver_radius=config['receiver_radius'],
        atm_k=config['atm_k'],
        sigma_sun_mrad=config['sigma_sun_mrad'],
        sigma_track_mrad=config['sigma_track_mrad'],
        sigma_slope_mrad=config['sigma_slope_mrad'],
        rho_mirror=config['rho_mirror'],
        f_soiling=config['f_soiling'],
        f_refl=config['f_refl'],
        f_opt=config['f_opt']
    )
    
    print(f"[Config] Torre: {cfg.tower_height:.2f} m, Campo: r=[{cfg.r_min:.1f}, {cfg.r_max:.1f}] m")
    print(f"[Config] Receptor radio: {cfg.receiver_radius:.3f} m")
    print(f"[Config] Óptica: ρ={cfg.rho_mirror:.3f}, soiling={cfg.f_soiling:.3f}, f_refl={cfg.f_refl:.3f}")
    
    (cmds, opt_pts, pred_pts, ts, eta_opt, eta_pred, eta_eff,
     den_series, sat_series, sing_series, rda_series) = plan_commands(
        start, end, config['step_s'], config['lat'], config['lon'], cfg, mode="heading_follow"
    )
    
    print(f"[OK] {len(opt_pts)} puntos calculados para trayectoria óptima")
    
    # Construir trayectoria circular (radio fijo = primer punto óptimo)
    if opt_pts:
        r0 = math.hypot(opt_pts[0][0], opt_pts[0][1])
        circ_pts = []
        for xo, yo in opt_pts:
            phi = math.atan2(yo, xo)
            xc = r0 * math.cos(phi)
            yc = r0 * math.sin(phi)
            circ_pts.append((xc, yc))
        print(f"[OK] Trayectoria circular construida con r0={r0:.3f} m")
    else:
        circ_pts = []
        print("[WARN] No hay puntos óptimos")
        return
    
    # Convertir a arrays numpy (trayectorias base)
    x_opt_base = np.array([p[0] for p in opt_pts])
    y_opt_base = np.array([p[1] for p in opt_pts])
    t_opt_absolute = np.array([t.hour + t.minute/60.0 + t.second/3600.0 for t in ts])
    
    x_circ_base = np.array([p[0] for p in circ_pts])
    y_circ_base = np.array([p[1] for p in circ_pts])
    t_circ_absolute = t_opt_absolute.copy()
    
    # ========================================================================
    # APLICAR CONFIGURACIÓN DE ENJAMBRE (si está activado)
    # ========================================================================
    use_swarm = config.get('use_swarm', False)
    
    if use_swarm:
        print("\n" + "="*70)
        print("CONFIGURACIÓN DE ENJAMBRE ACTIVADA")
        print("="*70)
        
        swarm_separation = float(config.get('swarm_separation', 1.0))
        swarm_offsets = get_hexagonal_swarm_offsets(swarm_separation)
        
        print(f"Heliostat en formación hexagonal: 1 central + 6 alrededor")
        print(f"Separación centro-a-centro: {swarm_separation:.3f} m")
        print(f"Total heliostat: {len(swarm_offsets)}")
        
        # Aplicar offsets a ambas trayectorias
        swarm_opt_trajectories = apply_swarm_to_trajectory(x_opt_base, y_opt_base, swarm_offsets)
        swarm_circ_trajectories = apply_swarm_to_trajectory(x_circ_base, y_circ_base, swarm_offsets)
        
        print(f"[OK] Trayectorias del enjambre generadas")
        
        # Para compatibilidad con código existente, usar el central como referencia
        x_opt = swarm_opt_trajectories[0][0]
        y_opt = swarm_opt_trajectories[0][1]
        x_circ = swarm_circ_trajectories[0][0]
        y_circ = swarm_circ_trajectories[0][1]
    else:
        print("\n[INFO] Modo heliostat único (sin enjambre)")
        swarm_offsets = [(0.0, 0.0)]  # Solo heliostat central
        swarm_opt_trajectories = [(x_opt_base, y_opt_base)]
        swarm_circ_trajectories = [(x_circ_base, y_circ_base)]
        
        x_opt = x_opt_base
        y_opt = y_opt_base
        x_circ = x_circ_base
        y_circ = y_circ_base
    
    # Información de debug sobre rangos temporales
    print(f"\n[DEBUG] Rango trayectorias: {t_opt_absolute.min():.2f}h → {t_opt_absolute.max():.2f}h")
    
    # ========================================================================
    # PASO 3: Integración con SolarPILOT (si está habilitado)
    # ========================================================================
    
    sol_opt_results = None
    sol_circ_results = None
    my_opt_results = None
    my_circ_results = None
    
    if use_sp:
        try:
            print("\n" + "="*70)
            print("PASO 3: Inicializando SolarPILOT")
            print("="*70)
            
            # Cambiar al directorio correcto donde está solarpilot.dll
            import os
            original_cwd = os.getcwd()
            script_dir = Path(__file__).resolve().parent
            # El DLL debería estar en SolarPILOT/deploy/api/
            os.chdir(script_dir)
            print(f"[SolarPILOT] Directorio de trabajo: {os.getcwd()}")
            
            from copylot import CoPylot
            cp = CoPylot()
            r = cp.data_create()
            
            # Configurar caso con parámetros sincronizados de la GUI
            print("[SolarPILOT] Configurando caso...")
            cp.data_set_string(r, "ambient.0.weather_file", config['weather_file'])
            cp.data_set_number(r, "solarfield.0.q_des", config['q_des'])
            cp.data_set_string(r, "receiver.0.rec_type", config['rec_type'])
            cp.data_set_string(r, "solarfield.0.des_sim_detail", "Single simulation point")
            
            # Sincronizar parámetros geométricos
            cp.data_set_number(r, "solarfield.0.tht", config['tower_height'])
            cp.data_set_number(r, "heliostat.0.width", config['hel_width'])
            cp.data_set_number(r, "heliostat.0.height", config['hel_height'])
            cp.data_set_number(r, "receiver.0.rec_height", config['rec_height'])
            cp.data_set_number(r, "receiver.0.rec_width", config['rec_width'])
            
            # Intentar sincronizar parámetros ópticos (si la API lo permite)
            try:
                # Reflectividad del espejo [heliostat.0.reflectance o heliostat.0.reflect]
                cp.data_set_number(r, "heliostat.0.reflectance", config['rho_mirror'])
                print(f"  ✓ Reflectividad configurada: {config['rho_mirror']:.3f}")
            except Exception:
                print(f"  [WARN] No se pudo configurar reflectividad en SP (usando default)")
            
            try:
                # Soiling (ensuciamiento) [heliostat.0.soiling]
                cp.data_set_number(r, "heliostat.0.soiling", config['f_soiling'])
                print(f"  ✓ Soiling configurado: {config['f_soiling']:.3f}")
            except Exception:
                print(f"  [WARN] No se pudo configurar soiling en SP (usando default)")
            
            print(f"\n[Sync Check] Parámetros configurados:")
            print(f"  Torre: {config['tower_height']:.2f} m")
            print(f"  Receptor: {config['rec_height']:.2f} x {config['rec_width']:.2f} m")
            print(f"  Heliostat: {config['hel_width']:.3f} x {config['hel_height']:.3f} m")
            print(f"  Modelo propio usa receiver_radius={config['receiver_radius']:.3f} m")
            
            print("[SolarPILOT] Generando layout...")
            cp.generate_layout(r)
            
            # Primera simulación para obtener geometría
            cp.data_set_number(r, "fluxsim.0.flux_month", day0.month)
            cp.data_set_number(r, "fluxsim.0.flux_day", day0.day)
            cp.data_set_number(r, "fluxsim.0.flux_hour", float(t_opt_absolute[0]))
            cp.simulate(r)
            
            mat, header = cp.detail_results(r, restype="matrix")
            H = [h.strip() for h in header]
            rows = [{H[j]: row[j] for j in range(len(H))} for row in mat]
            
            # Extraer geometría
            idx_xy = {}
            for rw in rows:
                try:
                    hid = rw.get("id", rw.get("heliostat_id"))
                    x = float(rw["x_location"])
                    y = float(rw["y_location"])
                    idx_xy[hid] = (x, y)
                except:
                    continue
            
            ids_list = list(idx_xy.keys())
            XY = np.array([idx_xy[i] for i in ids_list], dtype=float)
            print(f"[SolarPILOT] Layout: {len(ids_list)} heliostatos")
            
            # Crear lista de tiempos para evaluación SolarPILOT
            h_start = t_opt_absolute.min()
            h_end = t_opt_absolute.max()
            samples_per_hour = config['sp_samples']
            
            sp_times = []
            h = float(np.floor(h_start))
            while h < h_end - 1e-9:
                t_ini = max(h_start, h)
                t_fin = min(h_end, h + 1.0)
                if t_fin > t_ini:
                    dt = (t_fin - t_ini) / samples_per_hour
                    sp_times += [t_ini + (k + 0.5) * dt for k in range(samples_per_hour)]
                h += 1.0
            
            print(f"[SolarPILOT] Se evaluarán {len(sp_times)} puntos temporales")
            print(f"[DEBUG] Rango sp_times: {min(sp_times):.2f}h → {max(sp_times):.2f}h")
            print(f"[DEBUG] Primeros 3 sp_times: {sp_times[:3]}")
            print(f"[DEBUG] Últimos 3 sp_times: {sp_times[-3:]}")
            
            # Precalcular para óptima
            print("\n--- Trayectoria ÓPTIMA (SolarPILOT) ---")
            sol_opt_results = precompute_solarpilot_for_traj(
                cp, r, sp_times, x_opt, y_opt, t_opt_absolute,
                XY, ids_list, day0.month, day0.day
            )
            
            # Precalcular para circular
            print("\n--- Trayectoria CIRCULAR (SolarPILOT) ---")
            sol_circ_results = precompute_solarpilot_for_traj(
                cp, r, sp_times, x_circ, y_circ, t_circ_absolute,
                XY, ids_list, day0.month, day0.day
            )
            
            cp.data_free(r)
            # Restaurar directorio original
            os.chdir(original_cwd)
            print("\n[OK] SolarPILOT: Precálculo completado")
            
            # ================================================================
            # PASO 4: Calcular modelo propio para mismos tiempos
            # ================================================================
            print("\n" + "="*70)
            print("PASO 4: Calculando modelo propio para comparación")
            print("="*70)
            
            if use_swarm:
                # Calcular para todo el enjambre
                print("\n--- Enjambre ÓPTIMO (Modelo Propio) ---")
                swarm_opt_results = compute_swarm_model_series(
                    sp_times, swarm_opt_trajectories, t_opt_absolute, cfg, config['lat'], config['lon'], day0
                )
                
                print("\n--- Enjambre CIRCULAR (Modelo Propio) ---")
                swarm_circ_results = compute_swarm_model_series(
                    sp_times, swarm_circ_trajectories, t_circ_absolute, cfg, config['lat'], config['lon'], day0
                )
                
                # Para compatibilidad con código de exportación, usar heliostat central
                my_opt_results = swarm_opt_results[0]
                my_circ_results = swarm_circ_results[0]
                
                # Guardar resultados completos del enjambre para uso posterior
                my_opt_results['swarm_results'] = swarm_opt_results
                my_circ_results['swarm_results'] = swarm_circ_results
                
                # Calcular eficiencia máxima del enjambre en cada timestep
                n_times = len(sp_times)
                max_eta_opt = np.zeros(n_times)
                max_eta_circ = np.zeros(n_times)
                
                for t_idx in range(n_times):
                    # Extraer eficiencias de todos los heliostat en este timestep
                    etas_opt = [swarm_opt_results[h]['eta_total'][t_idx] for h in range(len(swarm_offsets))]
                    etas_circ = [swarm_circ_results[h]['eta_total'][t_idx] for h in range(len(swarm_offsets))]
                    
                    # Guardar el máximo (heliostat más eficiente)
                    max_eta_opt[t_idx] = np.nanmax(etas_opt) if not all(np.isnan(etas_opt)) else np.nan
                    max_eta_circ[t_idx] = np.nanmax(etas_circ) if not all(np.isnan(etas_circ)) else np.nan
                
                # Guardar para uso en animación
                my_opt_results['max_eta_swarm'] = max_eta_opt
                my_circ_results['max_eta_swarm'] = max_eta_circ
                
                print(f"\n[ENJAMBRE] Eficiencia máxima calculada:")
                print(f"  Óptima:   η̄_max = {np.nanmean(max_eta_opt):.4f}")
                print(f"  Circular: η̄_max = {np.nanmean(max_eta_circ):.4f}")
            else:
                # Modo heliostat único
                print("\n--- Trayectoria ÓPTIMA (Modelo Propio) ---")
                my_opt_results = compute_my_model_series(
                    sp_times, x_opt, y_opt, t_opt_absolute, cfg, config['lat'], config['lon'], day0
                )
                
                print("\n--- Trayectoria CIRCULAR (Modelo Propio) ---")
                my_circ_results = compute_my_model_series(
                    sp_times, x_circ, y_circ, t_circ_absolute, cfg, config['lat'], config['lon'], day0
                )
            
        except Exception as e:
            print(f"\n[ERROR] Fallo en integración SolarPILOT: {e}")
            print("Continuando sin SolarPILOT...")
            import traceback
            traceback.print_exc()
            use_sp = False
    
    # ========================================================================
    # PASO 5: Exportar resultados
    # ========================================================================
    print("\n" + "="*70)
    print("PASO 5: Exportando resultados")
    print("="*70)
    
    exp = Path(config['out_dir'])
    exp.mkdir(exist_ok=True)
    
    if use_sp and sol_opt_results is not None and my_opt_results is not None:
        # CSV para óptima
        with open(exp / "results_optima.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "x", "y", "sol_id", "my_eta", "sol_eta_no_sb", 
                        "diff", "my_eta_cos", "my_eta_att", "my_eta_int", "my_dist"])
            
            for i, t_h in enumerate(sp_times):
                px, py = nearest_traj_point(t_h, x_opt, y_opt, t_opt_absolute)
                sol_eta_nsb = sol_opt_results["eta_no_sb"][i]
                sol_id = sol_opt_results["sol_id"][i]
                my_eta = my_opt_results["eta_total"][i]
                diff = my_eta - sol_eta_nsb
                
                w.writerow([
                    f"{t_h:.4f}", f"{px:.6f}", f"{py:.6f}", sol_id,
                    f"{my_eta:.6f}", f"{sol_eta_nsb:.6f}",
                    f"{diff:.6f}",
                    f"{my_opt_results['eta_cos'][i]:.6f}",
                    f"{my_opt_results['eta_att'][i]:.6f}",
                    f"{my_opt_results['eta_int'][i]:.6f}",
                    f"{my_opt_results['dist'][i]:.3f}"
                ])
        print(f"✓ {exp / 'results_optima.csv'}")
        
        # CSV para circular
        with open(exp / "results_circular.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "x", "y", "sol_id", "my_eta", "sol_eta_no_sb",
                        "diff", "my_eta_cos", "my_eta_att", "my_eta_int", "my_dist"])
            
            for i, t_h in enumerate(sp_times):
                px, py = nearest_traj_point(t_h, x_circ, y_circ, t_circ_absolute)
                sol_eta_nsb = sol_circ_results["eta_no_sb"][i]
                sol_id = sol_circ_results["sol_id"][i]
                my_eta = my_circ_results["eta_total"][i]
                diff = my_eta - sol_eta_nsb
                
                w.writerow([
                    f"{t_h:.4f}", f"{px:.6f}", f"{py:.6f}", sol_id,
                    f"{my_eta:.6f}", f"{sol_eta_nsb:.6f}",
                    f"{diff:.6f}",
                    f"{my_circ_results['eta_cos'][i]:.6f}",
                    f"{my_circ_results['eta_att'][i]:.6f}",
                    f"{my_circ_results['eta_int'][i]:.6f}",
                    f"{my_circ_results['dist'][i]:.3f}"
                ])
        print(f"✓ {exp / 'results_circular.csv'}")
        
        # ====================================================================
        # Estadísticas
        # ====================================================================
        print("\n" + "="*70)
        print("ESTADÍSTICAS COMPARATIVAS")
        print("="*70)
        
        if use_swarm:
            print(f"\n[ENJAMBRE] Configuración: {len(swarm_offsets)} heliostat")
            print(f"           Separación: {float(config.get('swarm_separation', 1.0)):.3f} m")
            print(f"           Posiciones offsets: {swarm_offsets[:3]}...")
        
        def mean_valid(arr):
            arr = np.array(arr)
            arr = arr[~np.isnan(arr)]
            return float(np.mean(arr)) if len(arr) > 0 else np.nan
        
        opt_my_mean = mean_valid(my_opt_results["eta_total"])
        opt_sol_nsb_mean = mean_valid(sol_opt_results["eta_no_sb"])
        opt_fixed_mean = mean_valid(sol_opt_results["eta_fixed"])
        
        circ_my_mean = mean_valid(my_circ_results["eta_total"])
        circ_sol_nsb_mean = mean_valid(sol_circ_results["eta_no_sb"])
        circ_fixed_mean = mean_valid(sol_circ_results["eta_fixed"])
        
        print(f"\nTRAYECTORIA ÓPTIMA:")
        print(f"  Modelo propio:               η̄ = {opt_my_mean:.4f}")
        print(f"  SolarPILOT (trayectoria):    η̄ = {opt_sol_nsb_mean:.4f}")
        print(f"  SolarPILOT (fijo id={sol_opt_results.get('fixed_id', 'N/A')}):  η̄ = {opt_fixed_mean:.4f}")
        print(f"  Diferencia (propio - SP):         {opt_my_mean - opt_sol_nsb_mean:+.4f}")
        print(f"  Diferencia (fijo - trayectoria):  {opt_fixed_mean - opt_sol_nsb_mean:+.4f}")
        
        print(f"\nTRAYECTORIA CIRCULAR:")
        print(f"  Modelo propio:               η̄ = {circ_my_mean:.4f}")
        print(f"  SolarPILOT (trayectoria):    η̄ = {circ_sol_nsb_mean:.4f}")
        print(f"  SolarPILOT (fijo id={sol_circ_results.get('fixed_id', 'N/A')}):  η̄ = {circ_fixed_mean:.4f}")
        print(f"  Diferencia (propio - SP):         {circ_my_mean - circ_sol_nsb_mean:+.4f}")
        print(f"  Diferencia (fijo - trayectoria):  {circ_fixed_mean - circ_sol_nsb_mean:+.4f}")
        print(f"  Diferencia (propio - SP):         {circ_my_mean - circ_sol_nsb_mean:+.4f}")
        
        # Análisis de NaN
        opt_my_nan = sum(1 for x in my_opt_results["eta_total"] if not np.isfinite(x))
        opt_sol_nan = sum(1 for x in sol_opt_results["eta_no_sb"] if not np.isfinite(x))
        circ_my_nan = sum(1 for x in my_circ_results["eta_total"] if not np.isfinite(x))
        circ_sol_nan = sum(1 for x in sol_circ_results["eta_no_sb"] if not np.isfinite(x))
        
        if opt_my_nan > 0 or opt_sol_nan > 0 or circ_my_nan > 0 or circ_sol_nan > 0:
            print(f"\n[INFO] Valores NaN/infinitos encontrados:")
            if opt_my_nan > 0:
                print(f"  Óptima (Propio):    {opt_my_nan}/{len(my_opt_results['eta_total'])}")
            if opt_sol_nan > 0:
                nan_times = [sp_times[i] for i, x in enumerate(sol_opt_results['eta_no_sb']) if not np.isfinite(x)]
                print(f"  Óptima (SolarPILOT): {opt_sol_nan}/{len(sol_opt_results['eta_no_sb'])}")
                print(f"    Tiempos con NaN: {[f'{t:.2f}h' for t in nan_times[:10]]}")  # Mostrar hasta 10
            if circ_my_nan > 0:
                print(f"  Circular (Propio):   {circ_my_nan}/{len(my_circ_results['eta_total'])}")
            if circ_sol_nan > 0:
                nan_times = [sp_times[i] for i, x in enumerate(sol_circ_results['eta_no_sb']) if not np.isfinite(x)]
                print(f"  Circular (SolarPILOT): {circ_sol_nan}/{len(sol_circ_results['eta_no_sb'])}")
                print(f"    Tiempos con NaN: {[f'{t:.2f}h' for t in nan_times[:10]]}")  # Mostrar hasta 10
        
        # Estadísticas del enjambre (si está activo)
        if use_swarm and 'swarm_results' in my_opt_results:
            print(f"\n[ENJAMBRE] Análisis de eficiencias por heliostat (óptima):")
            for hel_idx, hel_result in enumerate(my_opt_results['swarm_results']):
                hel_mean = mean_valid(hel_result['eta_total'])
                position_str = "Central" if hel_idx == 0 else f"#{hel_idx}"
                print(f"  Heliostat {position_str:8s}: η̄ = {hel_mean:.4f}")
            
            # Eficiencia promedio del enjambre completo
            all_etas = []
            for hel_result in my_opt_results['swarm_results']:
                all_etas.extend([e for e in hel_result['eta_total'] if np.isfinite(e)])
            swarm_mean = float(np.mean(all_etas)) if all_etas else np.nan
            print(f"  Enjambre completo:    η̄ = {swarm_mean:.4f}")
            
            # Mostrar heliostat más eficiente
            if 'max_eta_swarm' in my_opt_results:
                max_swarm_mean = mean_valid(my_opt_results['max_eta_swarm'])
                print(f"  Mejor en cada tiempo: η̄ = {max_swarm_mean:.4f}")
            
            print(f"\n[ENJAMBRE] Análisis de eficiencias por heliostat (circular):")
            for hel_idx, hel_result in enumerate(my_circ_results['swarm_results']):
                hel_mean = mean_valid(hel_result['eta_total'])
                position_str = "Central" if hel_idx == 0 else f"#{hel_idx}"
                print(f"  Heliostat {position_str:8s}: η̄ = {hel_mean:.4f}")
            
            all_etas_circ = []
            for hel_result in my_circ_results['swarm_results']:
                all_etas_circ.extend([e for e in hel_result['eta_total'] if np.isfinite(e)])
            swarm_mean_circ = float(np.mean(all_etas_circ)) if all_etas_circ else np.nan
            print(f"  Enjambre completo:    η̄ = {swarm_mean_circ:.4f}")
            
            if 'max_eta_swarm' in my_circ_results:
                max_swarm_circ_mean = mean_valid(my_circ_results['max_eta_swarm'])
                print(f"  Mejor en cada tiempo: η̄ = {max_swarm_circ_mean:.4f}")
        
        print("="*70)
    
    # ========================================================================
    # PASO 6: Animación
    # ========================================================================
    # Verificar si se debe generar animación (mostrar o guardar)
    should_animate = config.get('show_animation', False) or config.get('save_gif', False)
    
    if should_animate:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            print("\n" + "="*70)
            print("PASO 6: Generando animación comparativa")
            print("="*70)
            
            xs_opt = [p[0] for p in opt_pts]
            ys_opt = [p[1] for p in opt_pts]
            xs_circ = [p[0] for p in circ_pts]
            ys_circ = [p[1] for p in circ_pts]
            
            # Preparar trayectorias del enjambre para visualización
            if use_swarm:
                swarm_opt_xs = [[traj[0][i] for i in range(len(traj[0]))] for traj in swarm_opt_trajectories]
                swarm_opt_ys = [[traj[1][i] for i in range(len(traj[1]))] for traj in swarm_opt_trajectories]
                swarm_circ_xs = [[traj[0][i] for i in range(len(traj[0]))] for traj in swarm_circ_trajectories]
                swarm_circ_ys = [[traj[1][i] for i in range(len(traj[1]))] for traj in swarm_circ_trajectories]
            else:
                swarm_opt_xs = [xs_opt]
                swarm_opt_ys = [ys_opt]
                swarm_circ_xs = [xs_circ]
                swarm_circ_ys = [ys_circ]
            
            # Preparar datos para animación
            n_frames = len(sp_times)
            sp_times_arr = np.array(sp_times)
            
            # Configurar figura con 5 subplots en 2 filas
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.25, hspace=0.35)
            
            ax_traj = fig.add_subplot(gs[0, 0])
            ax_layout = fig.add_subplot(gs[0, 1])
            ax_eta_opt = fig.add_subplot(gs[1, 0])
            ax_eta_circ = fig.add_subplot(gs[1, 1])
            ax_eta_comp = fig.add_subplot(gs[1, 2])
            
            # Configurar subplot de trayectorias
            ax_traj.set_aspect('equal')
            ax_traj.axhline(0, lw=0.8, ls='--', c='#aaa')
            ax_traj.axvline(0, lw=0.8, ls='--', c='#aaa')
            ax_traj.scatter([0], [0], s=80, c='#d35400', zorder=3, marker='s', label='Torre')
            
            # Plotear trayectorias del enjambre
            if use_swarm:
                # Mostrar todas las trayectorias del enjambre
                for i, (xs, ys) in enumerate(zip(swarm_opt_xs, swarm_opt_ys)):
                    alpha = 0.8 if i == 0 else 0.3  # Central más visible
                    lw = 1.8 if i == 0 else 1.0
                    label = 'Tray. Óptima (Central)' if i == 0 else ('Tray. Óptima (Enjambre)' if i == 1 else None)
                    ax_traj.plot(xs, ys, lw=lw, c='tab:blue', alpha=alpha, label=label)
                
                for i, (xs, ys) in enumerate(zip(swarm_circ_xs, swarm_circ_ys)):
                    alpha = 0.8 if i == 0 else 0.3
                    lw = 1.8 if i == 0 else 1.0
                    label = 'Tray. Circular (Central)' if i == 0 else ('Tray. Circular (Enjambre)' if i == 1 else None)
                    ax_traj.plot(xs, ys, lw=lw, c='tab:green', alpha=alpha, label=label)
            else:
                ax_traj.plot(xs_opt, ys_opt, lw=1.5, c='tab:blue', alpha=0.4, label='Tray. Óptima')
                ax_traj.plot(xs_circ, ys_circ, lw=1.5, c='tab:green', alpha=0.4, label='Tray. Circular')
            
            ax_traj.set_xlabel('Este (m)')
            ax_traj.set_ylabel('Norte (m)')
            title_suffix = f' - Enjambre ({len(swarm_offsets)} hel.)' if use_swarm else ''
            ax_traj.set_title(f'Trayectorias{title_suffix}\\n{day0.strftime("%Y-%m-%d")}')
            ax_traj.legend(loc='upper right', fontsize=9)
            ax_traj.grid(alpha=0.3)
            
            # Elementos animados - trayectorias (uno por cada heliostat del enjambre)
            hel_opt_markers = []
            hel_circ_markers = []
            
            for i in range(len(swarm_offsets)):
                size_central = 12 if i == 0 else 8
                alpha_central = 1.0 if i == 0 else 0.6
                marker_opt, = ax_traj.plot([], [], 'o', color='tab:blue', ms=size_central, 
                                           alpha=alpha_central, zorder=5)
                marker_circ, = ax_traj.plot([], [], 's', color='tab:green', ms=size_central-2, 
                                            alpha=alpha_central, zorder=5)
                hel_opt_markers.append(marker_opt)
                hel_circ_markers.append(marker_circ)
            
            # Configurar subplot de layout SolarPILOT
            ax_layout.axhline(0, lw=0.8, ls='--', c='#aaa')
            ax_layout.axvline(0, lw=0.8, ls='--', c='#aaa')
            ax_layout.scatter([0], [0], s=80, c='#d35400', zorder=3, marker='s', label='Torre')
            # Plotear todos los heliostatos del layout
            ax_layout.scatter(XY[:, 0], XY[:, 1], s=8, c='lightgray', alpha=0.5, zorder=1)
            ax_layout.set_xlabel('Este (m)')
            ax_layout.set_ylabel('Norte (m)')
            ax_layout.set_title(f'Layout SolarPILOT\\n{len(XY)} heliostatos')
            ax_layout.grid(alpha=0.3)
            # Aplicar zoom en la zona de la torre (acercado para ver mejor los heliostatos seleccionados)
            ax_layout.set_xlim(-10, 10)
            ax_layout.set_ylim(-10, 10)
            ax_layout.set_aspect('equal', adjustable='box')
            
            # Elementos animados - heliostatos seleccionados en layout
            hel_layout_opt, = ax_layout.plot([], [], 'o', color='tab:blue', ms=10, 
                                             markeredgecolor='black', markeredgewidth=1.5,
                                             zorder=5, label='Selec. Óptima')
            hel_layout_circ, = ax_layout.plot([], [], 's', color='tab:green', ms=8,
                                              markeredgecolor='black', markeredgewidth=1.5,
                                              zorder=4, label='Selec. Circular')
            ax_layout.legend(loc='upper right', fontsize=9)
            
            # Configurar subplot eficiencias óptima
            ax_eta_opt.set_xlim(sp_times_arr[0], sp_times_arr[-1])
            ax_eta_opt.set_ylim(0.8, 1.05)
            ax_eta_opt.set_xlabel('Hora decimal [h]', fontsize=10)
            ax_eta_opt.set_ylabel('Eficiencia η', fontsize=10)
            ax_eta_opt.set_title('TRAYECTORIA ÓPTIMA', fontsize=11, fontweight='bold')
            ax_eta_opt.grid(alpha=0.3)
            
            # Configurar subplot eficiencias circular
            ax_eta_circ.set_xlim(sp_times_arr[0], sp_times_arr[-1])
            ax_eta_circ.set_ylim(0.8, 1.05)
            ax_eta_circ.set_xlabel('Hora decimal [h]', fontsize=10)
            ax_eta_circ.set_ylabel('Eficiencia η', fontsize=10)
            ax_eta_circ.set_title('TRAYECTORIA CIRCULAR', fontsize=11, fontweight='bold')
            ax_eta_circ.grid(alpha=0.3)
            
            # Configurar subplot comparación óptima vs circular
            ax_eta_comp.set_xlim(sp_times_arr[0], sp_times_arr[-1])
            ax_eta_comp.set_ylim(0.8, 1.05)
            ax_eta_comp.set_xlabel('Hora decimal [h]', fontsize=10)
            ax_eta_comp.set_ylabel('Eficiencia η', fontsize=10)
            ax_eta_comp.set_title('COMPARACIÓN TRAYECTORIAS', fontsize=11, fontweight='bold')
            ax_eta_comp.grid(alpha=0.3)
            
            # Líneas de eficiencia (sin marcadores)
            if use_sp and sol_opt_results is not None:
                line_my_opt, = ax_eta_opt.plot([], [], '-', label='Modelo propio (central)', 
                                                lw=2, color='tab:blue', alpha=0.9)
                line_sol_opt, = ax_eta_opt.plot([], [], '-', label='SolarPILOT (trayectoria)', 
                                                 lw=2, color='tab:red', alpha=0.7)
                line_fixed_opt, = ax_eta_opt.plot([], [], '-.', label=f'SP Fijo (η̄={sol_opt_results.get("fixed_mean_eta", 0):.3f})',
                                                   lw=1.5, color='purple', alpha=0.8)
                
                line_my_circ, = ax_eta_circ.plot([], [], '-', label='Modelo propio (central)',
                                                  lw=2, color='tab:green', alpha=0.9)
                line_sol_circ, = ax_eta_circ.plot([], [], '-', label='SolarPILOT (trayectoria)',
                                                   lw=2, color='tab:red', alpha=0.7)
                line_fixed_circ, = ax_eta_circ.plot([], [], '-.', label=f'SP Fijo (η̄={sol_circ_results.get("fixed_mean_eta", 0):.3f})',
                                                     lw=1.5, color='purple', alpha=0.8)
                
                # Líneas para heliostat más eficiente del enjambre (si aplica)
                if use_swarm:
                    line_max_opt, = ax_eta_opt.plot([], [], '-.', label='Mejor enjambre',
                                                     lw=1.5, color='orange', alpha=0.8)
                    line_max_circ, = ax_eta_circ.plot([], [], '-.', label='Mejor enjambre',
                                                       lw=1.5, color='orange', alpha=0.8)
                else:
                    line_max_opt, line_max_circ = None, None
                
                # Comparación: óptima vs circular (ambos modelos)
                line_comp_opt_my, = ax_eta_comp.plot([], [], '-', label='Óptima (propio)',
                                                      lw=2, color='tab:blue', alpha=0.9)
                line_comp_circ_my, = ax_eta_comp.plot([], [], '-', label='Circular (propio)',
                                                       lw=2, color='tab:green', alpha=0.9)
                line_comp_opt_sp, = ax_eta_comp.plot([], [], '--', label='Óptima (SP)',
                                                      lw=2, color='tab:blue', alpha=0.6)
                line_comp_circ_sp, = ax_eta_comp.plot([], [], '--', label='Circular (SP)',
                                                       lw=2, color='tab:green', alpha=0.6)
                
                ax_eta_opt.legend(loc='lower right', fontsize=9)
                ax_eta_circ.legend(loc='lower right', fontsize=9)
                ax_eta_comp.legend(loc='lower right', fontsize=9)
                
                # Texto con valores
                text_opt = ax_eta_opt.text(0.02, 0.98, '', transform=ax_eta_opt.transAxes,
                                           fontsize=9, va='top', family='monospace',
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                text_circ = ax_eta_circ.text(0.02, 0.98, '', transform=ax_eta_circ.transAxes,
                                             fontsize=9, va='top', family='monospace',
                                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                text_comp = ax_eta_comp.text(0.02, 0.98, '', transform=ax_eta_comp.transAxes,
                                             fontsize=9, va='top', family='monospace',
                                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            def init():
                for marker in hel_opt_markers:
                    marker.set_data([], [])
                for marker in hel_circ_markers:
                    marker.set_data([], [])
                hel_layout_opt.set_data([], [])
                hel_layout_circ.set_data([], [])
                if use_sp and sol_opt_results is not None:
                    line_my_opt.set_data([], [])
                    line_sol_opt.set_data([], [])
                    line_fixed_opt.set_data([], [])
                    if use_swarm and line_max_opt is not None:
                        line_max_opt.set_data([], [])
                        line_max_circ.set_data([], [])
                    line_my_circ.set_data([], [])
                    line_sol_circ.set_data([], [])
                    line_fixed_circ.set_data([], [])
                    line_comp_opt_my.set_data([], [])
                    line_comp_circ_my.set_data([], [])
                    line_comp_opt_sp.set_data([], [])
                    line_comp_circ_sp.set_data([], [])
                    text_opt.set_text('')
                    text_circ.set_text('')
                    text_comp.set_text('')
                    return (*hel_opt_markers, *hel_circ_markers, hel_layout_opt, hel_layout_circ, 
                            line_my_opt, line_sol_opt, line_my_circ, line_sol_circ, 
                            line_comp_opt_my, line_comp_circ_my, line_comp_opt_sp, line_comp_circ_sp,
                            text_opt, text_circ, text_comp)
                return (*hel_opt_markers, *hel_circ_markers, hel_layout_opt, hel_layout_circ)
            
            def animate(frame):
                # Actualizar posición de todos los heliostat del enjambre
                t_h = sp_times[frame]
                
                # Actualizar cada heliostat del enjambre
                for hel_idx in range(len(swarm_offsets)):
                    # Óptima
                    x_traj_opt, y_traj_opt = swarm_opt_trajectories[hel_idx]
                    px_opt, py_opt = nearest_traj_point(t_h, x_traj_opt, y_traj_opt, t_opt_absolute)
                    
                    if px_opt is not None and py_opt is not None:
                        hel_opt_markers[hel_idx].set_data([px_opt], [py_opt])
                    else:
                        hel_opt_markers[hel_idx].set_data([], [])
                    
                    # Circular
                    x_traj_circ, y_traj_circ = swarm_circ_trajectories[hel_idx]
                    px_circ, py_circ = nearest_traj_point(t_h, x_traj_circ, y_traj_circ, t_circ_absolute)
                    
                    if px_circ is not None and py_circ is not None:
                        hel_circ_markers[hel_idx].set_data([px_circ], [py_circ])
                    else:
                        hel_circ_markers[hel_idx].set_data([], [])
                
                # Usar heliostat central para referencias
                px_opt, py_opt = nearest_traj_point(t_h, x_opt, y_opt, t_opt_absolute)
                px_circ, py_circ = nearest_traj_point(t_h, x_circ, y_circ, t_circ_absolute)
                
                # Actualizar heliostatos seleccionados en layout
                if use_sp and sol_opt_results is not None:
                    # Encontrar heliostatos más cercanos a las posiciones actuales
                    hel_positions_opt = []
                    hel_positions_circ = []
                    
                    # Mostrar en el layout los heliostatos fijos seleccionados por SolarPILOT
                    fixed_opt_id = sol_opt_results.get('fixed_id') if sol_opt_results else None
                    fixed_circ_id = sol_circ_results.get('fixed_id') if sol_circ_results else None

                    if fixed_opt_id is not None and fixed_opt_id in ids_list:
                        idx_opt = ids_list.index(fixed_opt_id)
                        hel_positions_opt = [XY[idx_opt, 0]], [XY[idx_opt, 1]]

                    if fixed_circ_id is not None and fixed_circ_id in ids_list:
                        idx_circ = ids_list.index(fixed_circ_id)
                        hel_positions_circ = [XY[idx_circ, 0]], [XY[idx_circ, 1]]
                    
                    hel_layout_opt.set_data(hel_positions_opt if hel_positions_opt else ([], []))
                    hel_layout_circ.set_data(hel_positions_circ if hel_positions_circ else ([], []))
                
                if use_sp and sol_opt_results is not None:
                    # Actualizar curvas de eficiencia (hasta frame actual)
                    times_so_far = sp_times_arr[:frame+1]
                    
                    my_opt_so_far = my_opt_results["eta_total"][:frame+1]
                    sol_opt_so_far = sol_opt_results["eta_no_sb"][:frame+1]
                    
                    my_circ_so_far = my_circ_results["eta_total"][:frame+1]
                    sol_circ_so_far = sol_circ_results["eta_no_sb"][:frame+1]
                    
                    line_my_opt.set_data(times_so_far, my_opt_so_far)
                    line_sol_opt.set_data(times_so_far, sol_opt_so_far)
                    line_my_circ.set_data(times_so_far, my_circ_so_far)
                    line_sol_circ.set_data(times_so_far, sol_circ_so_far)
                    
                    # Actualizar líneas del heliostat fijo
                    fixed_opt_so_far = sol_opt_results["eta_fixed"][:frame+1]
                    fixed_circ_so_far = sol_circ_results["eta_fixed"][:frame+1]
                    line_fixed_opt.set_data(times_so_far, fixed_opt_so_far)
                    line_fixed_circ.set_data(times_so_far, fixed_circ_so_far)
                    
                    # Actualizar líneas del heliostat más eficiente del enjambre
                    if use_swarm and line_max_opt is not None:
                        max_opt_so_far = my_opt_results['max_eta_swarm'][:frame+1]
                        max_circ_so_far = my_circ_results['max_eta_swarm'][:frame+1]
                        line_max_opt.set_data(times_so_far, max_opt_so_far)
                        line_max_circ.set_data(times_so_far, max_circ_so_far)
                    
                    # Actualizar comparación óptima vs circular (ambos modelos)
                    line_comp_opt_my.set_data(times_so_far, my_opt_so_far)
                    line_comp_circ_my.set_data(times_so_far, my_circ_so_far)
                    line_comp_opt_sp.set_data(times_so_far, sol_opt_so_far)
                    line_comp_circ_sp.set_data(times_so_far, sol_circ_so_far)
                    
                    # Actualizar texto con valores actuales
                    my_eta_opt_val = my_opt_results["eta_total"][frame]
                    sol_eta_opt_val = sol_opt_results["eta_no_sb"][frame]
                    diff_opt = my_eta_opt_val - sol_eta_opt_val
                    
                    my_eta_circ_val = my_circ_results["eta_total"][frame]
                    sol_eta_circ_val = sol_circ_results["eta_no_sb"][frame]
                    diff_circ = my_eta_circ_val - sol_eta_circ_val
                    
                    # Diferencia entre trayectorias
                    diff_traj = my_eta_opt_val - my_eta_circ_val
                    
                    # Construir textos con información del enjambre si aplica
                    if use_swarm:
                        max_eta_opt_val = my_opt_results['max_eta_swarm'][frame]
                        max_eta_circ_val = my_circ_results['max_eta_swarm'][frame]
                        
                        text_opt.set_text(
                            f't={t_h:.2f}h\\n'
                            f'η_central={my_eta_opt_val:.4f}\\n'
                            f'η_mejor={max_eta_opt_val:.4f}\\n'
                            f'η_SP={sol_eta_opt_val:.4f}\\n'
                            f'Δ(mejor-SP)={max_eta_opt_val-sol_eta_opt_val:+.4f}'
                        )
                        
                        text_circ.set_text(
                            f't={t_h:.2f}h\\n'
                            f'η_central={my_eta_circ_val:.4f}\\n'
                            f'η_mejor={max_eta_circ_val:.4f}\\n'
                            f'η_SP={sol_eta_circ_val:.4f}\\n'
                            f'Δ(mejor-SP)={max_eta_circ_val-sol_eta_circ_val:+.4f}'
                        )
                    else:
                        text_opt.set_text(
                            f't={t_h:.2f}h\\n'
                            f'η_mi={my_eta_opt_val:.4f}\\n'
                            f'η_SP={sol_eta_opt_val:.4f}\\n'
                            f'Δ={diff_opt:+.4f}'
                        )
                        
                        text_circ.set_text(
                            f't={t_h:.2f}h\\n'
                            f'η_mi={my_eta_circ_val:.4f}\\n'
                            f'η_SP={sol_eta_circ_val:.4f}\\n'
                            f'Δ={diff_circ:+.4f}'
                        )
                    
                    text_comp.set_text(
                        f't={t_h:.2f}h\\n'
                        f'η_opt_mi={my_eta_opt_val:.4f}\\n'
                        f'η_opt_SP={sol_eta_opt_val:.4f}\\n'
                        f'η_circ_mi={my_eta_circ_val:.4f}\\n'
                        f'η_circ_SP={sol_eta_circ_val:.4f}'
                    )
                    
                    return (*hel_opt_markers, *hel_circ_markers, hel_layout_opt, hel_layout_circ,
                            line_my_opt, line_sol_opt, line_my_circ, line_sol_circ, 
                            line_comp_opt_my, line_comp_circ_my, line_comp_opt_sp, line_comp_circ_sp,
                            text_opt, text_circ, text_comp)
                
                return (*hel_opt_markers, *hel_circ_markers, hel_layout_opt, hel_layout_circ)
            
            swarm_title = f' - Enjambre {len(swarm_offsets)} Heliostat' if use_swarm else ''
            plt.suptitle(f'Comparación: Modelo Propio vs SolarPILOT{swarm_title}', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            # Crear animación
            anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                 interval=200, blit=True, repeat=True)
            
            plt.tight_layout()
            
            # Exportar o mostrar animación según configuración
            if config.get('save_gif', False):
                gif_path = exp / 'animacion_comparacion.gif'
                print(f"[INFO] Guardando animación en: {gif_path}")
                print(f"       Esto puede tardar unos minutos con {n_frames} frames...")
                try:
                    anim.save(str(gif_path), writer='pillow', fps=config.get('gif_fps', 5), dpi=100)
                    print(f"✓ Animación guardada exitosamente: {gif_path}")
                except Exception as e:
                    print(f"[WARN] No se pudo guardar el GIF: {e}")
                    print("       Asegúrate de tener instalado 'pillow': pip install pillow")
            
            if config.get('show_animation', False):
                print(f"✓ Mostrando animación con {n_frames} frames")
                print("  (Cierra la ventana para continuar)")
                plt.show()
            elif not config.get('save_gif', False):
                print("[INFO] No se configuró mostrar ni guardar animación")
            
        except Exception as ex:
            print(f"[WARN] No se pudo generar animación: {ex}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"Resultados exportados en: {exp.absolute()}")


if __name__ == '__main__':
    main()
