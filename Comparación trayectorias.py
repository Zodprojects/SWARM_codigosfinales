# -*- coding: utf-8 -*-
"""
Versión modificada: sin 'heading' ni 'optima', solo trayectoria CIRCULAR derivada de 'heading'.
Mantiene exportes (CSV/SPT), caché y plots centrados en la trayectoria circular y un fijo derivado.
"""

import os, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from copylot import CoPylot

# ============ CONFIG ============
WEATHER  = r"C:\Users\Diego\Desktop\UNIVERSIDAD UPM\SWARM\CoPylot_Test\SolarPILOT\deploy\climate_files\Spain ESP Madrid (TMYx) - SolarPILOT.csv"
Q_DES    = 100.0
REC_TYPE = "Flat plate"
MONTH, DAY = 10, 28
H_START, H_END = 8, 18
SAMPLES_PER_HOUR = 10

TRJ_HEAD = r"C:\Users\Diego\Desktop\UNIVERSIDAD UPM\SWARM\CoPylot_Test\plan_eff.csv"
TRJ_OPT  = r"C:\Users\Diego\Desktop\UNIVERSIDAD UPM\SWARM\CoPylot_Test\plan_opt.csv"

PLOT = True

# shading/blocking ya vienen como FACTORES (0..1)
BLOCK_SHAD_ARE_LOSSES = False

# =================================

# ---------- columnas DR ----------
COL_ID          = "id"
COL_XLOC        = "x_location"
COL_YLOC        = "y_location"
COL_EFF_TOT     = "efficiency"
COL_COS         = "cosine"
COL_INT         = "intercept"
COL_REFL        = "reflectance"
COL_ATM         = "attenuation"
COL_BLOCK       = "blocking"
COL_SHAD        = "shading"
COL_CLOUD       = "clouds"

# ---------- utilidades ----------
def _n01(v, d=1.0):
    try:
        x = float(v)
    except:
        return d
    if 1.0 < x <= 100.0: x /= 100.0
    x = min(1.0, max(0.0, x))
    return float(x)

def eta_variants_from_row(row):
    """
    η_all = 'efficiency' del DR. 'shading' y 'blocking' son FACTORES (0..1).
    Para removerlos, dividimos por ellos (con protección por cero).
    """
    eta_all = _n01(row.get(COL_EFF_TOT, 0.0), d=0.0)
    sh_fac  = _n01(row.get(COL_SHAD,  1.0), d=1.0)
    bl_fac  = _n01(row.get(COL_BLOCK, 1.0), d=1.0)

    eps = 1e-8
    def safe_div(x, f):
        if not np.isfinite(f) or f < eps: return np.nan
        y = x / f
        return 1.0 if y > 1.0 else (0.0 if y < 0.0 else y)

    eta_no_shad = safe_div(eta_all, sh_fac)
    eta_no_blk  = safe_div(eta_all, bl_fac)
    eta_no_sb   = safe_div(eta_all, sh_fac * bl_fac if (np.isfinite(sh_fac) and np.isfinite(bl_fac)) else np.nan)

    return {
        "eta_all":     eta_all,
        "eta_no_shad": eta_no_shad,
        "eta_no_blk":  eta_no_blk,
        "eta_no_sb":   eta_no_sb,
        "factors": {
            "shading_raw": sh_fac, "blocking_raw": bl_fac,
            "cosine": _n01(row.get(COL_COS, 1.0)),
            "reflectance": _n01(row.get(COL_REFL, 1.0)),
            "attenuation": _n01(row.get(COL_ATM, 1.0)),
            "intercept": _n01(row.get(COL_INT, 1.0)),
            "clouds": _n01(row.get(COL_CLOUD, 1.0)),
            "efficiency_ref": eta_all,
        },
        "row": row
    }

def detail_rows(cp, r):
    mat, header = cp.detail_results(r, restype="matrix")
    H = [h.strip() for h in header]
    rows = [{H[j]: row[j] for j in range(len(H))} for row in mat]
    return rows, H

def coords_from_rows(rows):
    idx_xy = {}
    for rw in rows:
        try:
            hid = rw.get(COL_ID, rw.get("heliostat_id"))
            x = float(rw[COL_XLOC]); y = float(rw[COL_YLOC])
            idx_xy[hid] = (x, y)
        except:
            continue
    if not idx_xy:
        raise RuntimeError("No pude identificar columnas id/x_location/y_location.")
    return idx_xy

def load_traj_csv(path_csv, x_key="x", y_key="y", t_key="time"):
    xs, ys, hs = [], [], []
    with open(path_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row[x_key])); ys.append(float(row[y_key]))
            tv = row.get(t_key, "")
            if "T" in tv and ":" in tv:
                hh = int(tv[11:13]); mm = int(tv[14:16]); hs.append(hh+mm/60.0)
            elif ":" in tv:
                hh, mm = map(int, tv.split(":" )[:2]); hs.append(hh+mm/60.0)
            elif tv != "":
                hs.append(float(tv))
            else:
                hs.append(np.nan)
    return np.array(xs), np.array(ys), np.array(hs, dtype=float)

def nearest_traj_point(h_dec, tx, ty, th):
    mask = ~np.isnan(th)
    if mask.any():
        i = int(np.argmin(np.abs(th[mask] - h_dec)))
        idx = np.arange(len(th))[mask][i]
        return float(tx[idx]), float(ty[idx])
    mx, my = float(np.mean(tx)), float(np.mean(ty))
    d2 = (tx-mx)**2 + (ty-my)**2
    return float(tx[int(np.argmin(d2))]), float(ty[int(np.argmin(d2))])

def build_subtimes(h_start, h_end, samples_per_hour):
    times = []
    h = float(np.floor(h_start))
    while h < h_end - 1e-9:
        t_ini = max(h_start, h)
        t_fin = min(h_end,   h+1.0)
        if t_fin > t_ini:
            N = samples_per_hour
            dt = (t_fin - t_ini)/N
            times += [t_ini + (k+0.5)*dt for k in range(N)]
        h += 1.0
    return times

def detectar_huecos(t, y, thr_drop=0.08):
    t = np.asarray(t, float); y = np.asarray(y, float)
    dy = np.diff(y, prepend=y[0])
    idx = np.where(dy < -thr_drop)[0]
    idx = np.union1d(idx, np.where(y < 0.05)[0])
    return list(map(int, idx))

# ---- helpers robustos de escritura ----
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _unique_path(p: Path):
    if not p.exists():
        return p
    stem = p.stem; suf = p.suffix
    for k in range(1, 1000):
        q = p.with_name(f"{stem}__{k}{suf}")
        if not q.exists():
            return q
    return p.with_name(f"{stem}__tmp{np.random.randint(1e9)}{suf}")

def safe_write_csv(path: Path, header_list, row_values_list):
    _ensure_parent(path)
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_list is not None:
                w.writerow(header_list)
            for row_vals in row_values_list:
                w.writerow(row_vals)
        print(f"💾 CSV → {path}")
        return path
    except PermissionError:
        alt = _unique_path(path)
        with open(alt, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_list is not None:
                w.writerow(header_list)
            for row_vals in row_values_list:
                w.writerow(row_vals)
        print(f"⚠️ Archivo en uso. Guardado como → {alt}")
        return alt

# ---------- nueva utilidad: circunferencia desde heading ----------
def circular_from_heading(xh: np.ndarray, yh: np.ndarray):
    """Devuelve (xc, yc) imponiendo radio fijo r0 = ||(xh,yh)|| en el primer instante válido
       y conservando el ángulo φ_i = atan2(yh, xh) de heading para cada instante."""
    assert len(xh) == len(yh)
    mask = np.isfinite(xh) & np.isfinite(yh)
    if not mask.any():
        raise RuntimeError("No hay puntos válidos en la trayectoria heading para construir la circunferencia.")
    i0 = int(np.argmax(mask))  # primer True
    r0 = float(np.hypot(xh[i0], yh[i0]))
    phi = np.arctan2(yh, xh)
    xc = r0 * np.cos(phi)
    yc = r0 * np.sin(phi)
    return xc.astype(float), yc.astype(float), r0

# ========== MAIN ==========
def main():
    os.chdir(Path(__file__).resolve().parent)
    exp = Path("exports"); exp.mkdir(exist_ok=True)

    cp = CoPylot(); r = cp.data_create()

    # Caso
    assert cp.data_set_string(r, "ambient.0.weather_file", WEATHER)
    assert cp.data_set_number(r, "solarfield.0.q_des", Q_DES)
    assert cp.data_set_string(r, "receiver.0.rec_type", REC_TYPE)
    assert cp.data_set_string(r, "solarfield.0.des_sim_detail", "Single simulation point")
    assert cp.data_set_number(r, "solarfield.0.tht", 3)
    assert cp.data_set_number(r, "heliostat.0.width", 0.25)
    assert cp.data_set_number(r, "heliostat.0.height", 0.25)
    assert cp.data_set_number(r, "fluxsim.0.flux_month", MONTH)
    assert cp.data_set_number(r, "fluxsim.0.flux_day",   DAY)
    assert cp.data_set_number(r, "receiver.0.rec_height", 3)
    assert cp.data_set_number(r, "receiver.0.rec_width", 3)
    assert cp.generate_layout(r)

    # Trayectoria heading (solo se usa para derivar la circunferencia)
    xh, yh, th = load_traj_csv(TRJ_HEAD, "x","y","time")
    th += 1.0  # UTC→CET

    # Trayectoria óptima (se usa para comparar)
    xo, yo, to = load_traj_csv(TRJ_OPT, "x","y","time")
    to += 1.0  # UTC→CET

    # Trayectoria circular derivada de heading
    xc, yc, r0 = circular_from_heading(xh, yh)

    # Tiempos
    T = build_subtimes(H_START, H_END, SAMPLES_PER_HOUR)

    # Caché: t_sub -> (rows, header, by_id)
    results_cache = {}

    # Series (círculo + óptima) + fijos derivados
    series = {
        "circle":  {"eta_all":[], "eta_no_shad":[], "eta_no_blk":[], "eta_no_sb":[], "ids":[]},
        "optima":  {"eta_all":[], "eta_no_shad":[], "eta_no_blk":[], "eta_no_sb":[], "ids":[]},
        "fix_circle":  {"eta_all":[], "eta_no_shad":[], "eta_no_blk":[], "eta_no_sb":[]},
        "fix_opt":  {"eta_all":[], "eta_no_shad":[], "eta_no_blk":[], "eta_no_sb":[]},
    }

    # Para fijos y heatmap (media η_all)
    eta_all_acc = defaultdict(list)
    idx_xy = None; ids_list = None; XY = None
    ID_MAP_SNAPSHOT1 = None  # (idx_seq, id, x, y)

    for idx_t, t_sub in enumerate(T):
        cp.data_set_number(r, "fluxsim.0.flux_hour", float(t_sub))
        cp.simulate(r)
        rows, H = detail_rows(cp, r)

        by_id = {}
        for rw in rows:
            hid = rw.get(COL_ID, rw.get("heliostat_id"))
            if hid is None: continue
            ev = eta_variants_from_row(rw)
            by_id[hid] = ev
            eta_all_acc[hid].append(ev["eta_all"])

        results_cache[t_sub] = (rows, H, by_id)

        if idx_t == 0:
            # DR primera simulación (robusto a PermissionError)
            out_csv = exp / "dr_first_snapshot.csv"
            safe_write_csv(out_csv, H, [[rw.get(k, "") for k in H] for rw in rows])

            # SPT instante 0
            ts = f"{MONTH:02d}{DAY:02d}_{t_sub:.2f}h".replace(".", "p")
            spt_path = exp / f"case_{ts}.spt"
            if hasattr(cp, "save_from_script") and cp.save_from_script(r, str(spt_path)):
                print(f"📦 SPT → {spt_path}")
            else:
                print("⚠️ save_from_script no disponible en esta build")

            # Geometría & listas
            idx_xy = coords_from_rows(rows)
            ids_list = list(idx_xy.keys())
            XY = np.array([idx_xy[i] for i in ids_list], dtype=float)

            # Mapa idx secuencial <-> id <-> (x,y) para snapshot 1
            id_map = []
            for idx_seq, rw in enumerate(rows):
                hid = rw.get("id", rw.get("heliostat_id"))
                try:
                    x = float(rw["x_location"]); y = float(rw["y_location"])
                except:
                    x, y = float("nan"), float("nan")
                id_map.append((idx_seq, hid, x, y))
            ID_MAP_SNAPSHOT1 = id_map

            ids_only = [itm[1] for itm in id_map if itm[1] is not None]
            if ids_only:
                print(f"🧮 Snapshot #1: N={len(id_map)}  id_min={min(ids_only)}  id_max={max(ids_only)}")
            map_csv = exp / "id_map_snapshot1.csv"
            safe_write_csv(map_csv, ["idx_seq_snapshot","id_DR","x_location","y_location"],
                           [[a,b,f"{c:.3f}",f"{d:.3f}"] for (a,b,c,d) in id_map])

        # Círculo (mismo ángulo que heading, radio r0)
        # Reutilizamos el mismo índice temporal de heading (th) para la muestra más cercana:
        px_c, py_c = nearest_traj_point(t_sub, xc, yc, th)  # usamos 'th' (tiempos de heading)
        d2_c = (XY[:,0]-px_c)**2 + (XY[:,1]-py_c)**2
        id_c = ids_list[int(np.argmin(d2_c))]
        ev_c = by_id.get(id_c, None)

        for k in ["eta_all","eta_no_shad","eta_no_blk","eta_no_sb"]:
            series["circle"][k].append(np.nan  if ev_c is None else ev_c[k])
        series["circle"]["ids"].append(id_c)

        # Óptima (cercano a trayectoria óptima)
        px_o, py_o = nearest_traj_point(t_sub, xo, yo, to)
        d2_o = (XY[:,0]-px_o)**2 + (XY[:,1]-py_o)**2
        id_o = ids_list[int(np.argmin(d2_o))]
        ev_o = by_id.get(id_o, None)

        if abs(t_sub - 8.8) < 0.05 or abs(t_sub - 8.9) < 0.05:
            cos_o = ev_o["factors"]["cosine"] if ev_o else np.nan
            try:
                xy_o = XY[ids_list.index(id_o)]
            except Exception:
                xy_o = (np.nan, np.nan)
            print(f"[DEBUG óptima] t={t_sub:.2f}h  id={id_o}  XY={xy_o}  η={ev_o['eta_all']:.3f}  cos={cos_o:.3f}")

        for k in ["eta_all","eta_no_shad","eta_no_blk","eta_no_sb"]:
            series["optima"][k].append(np.nan if ev_o is None else ev_o[k])
        series["optima"]["ids"].append(id_o)

    # Fijo (mejor media η_all entre los usados de círculo)
    ids_circle = set(series["circle"]["ids"])
    best_circle_id, best_circle_eta = None, -1.0
    for hid in ids_circle:
        etas = eta_all_acc.get(hid, [])
        if not etas: continue
        m = float(np.mean(etas))
        if m > best_circle_eta: best_circle_eta, best_circle_id = m, hid

    print(f"⭐ Fijo CÍRCULO: id={best_circle_id}  η̄_all≈{best_circle_eta:.3f}")
    # Mejor fijo entre los usados por ÓPTIMA
    ids_opt = set(series["optima"]["ids"])
    best_opt_id, best_opt_eta = None, -1.0
    for hid in ids_opt:
        etas = eta_all_acc.get(hid, [])
        if not etas: continue
        m = float(np.mean(etas))
        if m > best_opt_eta: best_opt_eta, best_opt_id = m, hid

    print(f"⭐ Fijo ÓPTIMA:   id={best_opt_id}  η̄_all≈{best_opt_eta:.3f}")
    print(f"ⓘ Radio circular usado: r0 = {r0:.3f} m")

    # Series del fijo (solo circular y óptima)
    for t_sub in T:
        _, _, by_id = results_cache[t_sub]
        evc = by_id.get(best_circle_id, None)
        evo = by_id.get(best_opt_id, None)
        for k in ["eta_all","eta_no_shad","eta_no_blk","eta_no_sb"]:
            series["fix_circle"][k].append(np.nan if evc is None else evc[k])
            series["fix_opt"][k].append(np.nan if evo is None else evo[k])

    # ---- CSV extendido (círculo + óptima) ----
    csv_series = exp / "series_subhorarias_circle_optima_ext.csv"
    safe_write_csv(csv_series,
        ["t_sub",
         "etaC_all","etaC_noSh","etaC_noBl","etaC_noSB",
         "etaO_all","etaO_noSh","etaO_noBl","etaO_noSB",
         "etaFC_all","etaFC_noSh","etaFC_noBl","etaFC_noSB",
         "etaFO_all","etaFO_noSh","etaFO_noBl","etaFO_noSB",
         "id_circle","id_opt","id_fixed_circle","id_fixed_opt"],
        [[f"{t:.4f}",
          f"{series['circle']['eta_all'][i]:.6f}",
          f"{series['circle']['eta_no_shad'][i]:.6f}",
          f"{series['circle']['eta_no_blk'][i]:.6f}",
          f"{series['circle']['eta_no_sb'][i]:.6f}",
          f"{series['optima']['eta_all'][i]:.6f}",
          f"{series['optima']['eta_no_shad'][i]:.6f}",
          f"{series['optima']['eta_no_blk'][i]:.6f}",
          f"{series['optima']['eta_no_sb'][i]:.6f}",
          f"{series['fix_circle']['eta_all'][i]:.6f}",
          f"{series['fix_circle']['eta_no_shad'][i]:.6f}",
          f"{series['fix_circle']['eta_no_blk'][i]:.6f}",
          f"{series['fix_circle']['eta_no_sb'][i]:.6f}",
          f"{series['fix_opt']['eta_all'][i]:.6f}",
          f"{series['fix_opt']['eta_no_shad'][i]:.6f}",
          f"{series['fix_opt']['eta_no_blk'][i]:.6f}",
          f"{series['fix_opt']['eta_no_sb'][i]:.6f}",
          series["circle"]["ids"][i],
          series["optima"]["ids"][i],
          best_circle_id, best_opt_id] for i, t in enumerate(T)]
    )

    # ========== EXTRACCIÓN DR CÍRCULO — SIM 11 DE LA PRIMERA HORA ==========
    try:
        t_arr = np.array(T, float)
        idxs_first = [i for i, tt in enumerate(t_arr) if int(np.floor(tt)) == int(np.floor(H_START))]
        if not idxs_first:
            print("[DR CÍRCULO] No hay muestras en la primera hora.")
        else:
            if len(idxs_first) < 11:
                print(f"[DR CÍRCULO] Solo hay {len(idxs_first)} muestras en la primera hora. Uso la última.")
                k_global = idxs_first[-1]
            else:
                k_global = idxs_first[10]  # 11ª muestra

            t_target = T[k_global]
            rows, H, by_id = results_cache[t_target]

            if len(series["circle"]["ids"]) > k_global:
                id_c_k = series["circle"]["ids"][k_global]
            else:
                if (idx_xy is None) or (ids_list is None):
                    idx_xy = coords_from_rows(rows)
                    ids_list = list(idx_xy.keys())
                    XY = np.array([idx_xy[i] for i in ids_list], dtype=float)
                px_c, py_c = nearest_traj_point(t_target, xc, yc, th)
                d2 = (XY[:,0]-px_c)**2 + (XY[:,1]-py_c)**2
                id_c_k = ids_list[int(np.argmin(d2))]

            ev = by_id.get(id_c_k, None)
            if ev is None:
                print(f"[DR CÍRCULO] No hay by_id para t={t_target:.4f} id={id_c_k}.")
            else:
                rw = ev["row"]
                out_csv = exp / f"DR_circulo_t{t_target:.4f}_id{id_c_k}.csv"
                safe_write_csv(out_csv, H, [[rw.get(k, "") for k in H]])

                fc = ev["factors"]
                out_f = exp / f"DR_circulo_FACTORES_t{t_target:.4f}_id{id_c_k}.csv"
                safe_write_csv(out_f,
                    ["t_sub","id","eta_all","eta_no_shad","eta_no_blk","eta_no_sb",
                     "cosine","intercept","reflectance","attenuation","clouds",
                     "shading_factor","blocking_factor"],
                    [[f"{t_target:.4f}", id_c_k,
                      f"{ev['eta_all']:.6f}", f"{ev['eta_no_shad']:.6f}", f"{ev['eta_no_blk']:.6f}", f"{ev['eta_no_sb']:.6f}",
                      f"{fc.get('cosine',np.nan):.6f}", f"{fc.get('intercept',np.nan):.6f}",
                      f"{fc.get('reflectance',np.nan):.6f}", f"{fc.get('attenuation',np.nan):.6f}",
                      f"{fc.get('clouds',np.nan):.6f}", f"{fc.get('shading_raw',np.nan):.6f}", f"{fc.get('blocking_raw',np.nan):.6f}"]]
                )

                # SPT de ese instante (círculo)
                cp.data_set_number(r, "fluxsim.0.flux_hour", float(t_target))
                cp.simulate(r)
                ts = f"{MONTH:02d}{DAY:02d}_{t_target:.2f}h".replace(".", "p")
                spt_path = exp / f"circulo_t{ts}.spt"
                if hasattr(cp, "save_from_script") and cp.save_from_script(r, str(spt_path)):
                    print(f"📦 SPT (círculo, sim 11 primera hora) → {spt_path}")
    except Exception as e:
        print(f"[DR CÍRCULO] Error inesperado: {type(e).__name__}: {e}")

    # ---- PLOTS ----
    if PLOT:
        tA = np.array(T, float)

        # Círculo
        plt.figure(figsize=(10,4))
        plt.plot(tA, series["circle"]["eta_all"],     label="Círculo η (todo)",        marker="o", ms=3, lw=1)
        plt.plot(tA, series["circle"]["eta_no_shad"], label="Círculo η (sin sombras)", marker="o", ms=3, lw=1)
        plt.plot(tA, series["circle"]["eta_no_blk"],  label="Círculo η (sin bloqueos)",marker="o", ms=3, lw=1)
        plt.plot(tA, series["circle"]["eta_no_sb"],   label="Círculo η (sin somb/bloq)",marker="o", ms=3, lw=1)
        plt.ylim(0,1.05); plt.grid(alpha=.3); plt.legend()
        plt.xlabel("Hora decimal [h]"); plt.ylabel("η")
        plt.title(f"Círculo — η(t) variantes — {SAMPLES_PER_HOUR} muestras/h  (r0={r0:.2f} m)")
        plt.tight_layout(); plt.savefig(exp / "series_circle_variantes.png", dpi=160); plt.show()

        # Óptima
        plt.figure(figsize=(10,4))
        plt.plot(tA, series["optima"]["eta_all"],     label="Óptima η (todo)",        marker="o", ms=3, lw=1)
        plt.plot(tA, series["optima"]["eta_no_shad"], label="Óptima η (sin sombras)", marker="o", ms=3, lw=1)
        plt.plot(tA, series["optima"]["eta_no_blk"],  label="Óptima η (sin bloqueos)",marker="o", ms=3, lw=1)
        plt.plot(tA, series["optima"]["eta_no_sb"],   label="Óptima η (sin somb/bloq)",marker="o", ms=3, lw=1)
        plt.ylim(0,1.05); plt.grid(alpha=.3); plt.legend()
        plt.xlabel("Hora decimal [h]"); plt.ylabel("η")
        plt.title(f"Óptima — η(t) variantes — {SAMPLES_PER_HOUR} muestras/h")
        plt.tight_layout(); plt.savefig(exp / "series_optima_variantes.png", dpi=160); plt.show()

        # Comparativa "SIN SOMBRAS NI BLOQUEOS" (círculo y óptima y fijos)
        plt.figure(figsize=(10,4))
        plt.plot(tA, series["circle"]["eta_no_sb"],  label="Círculo (sin somb/bloq)",  marker="o", ms=3, lw=1)
        plt.plot(tA, series["optima"]["eta_no_sb"],  label="Óptima (sin somb/bloq)",  marker="o", ms=3, lw=1)
        plt.plot(tA, series["fix_circle"]["eta_no_sb"], label=f"Fijo CÍRCULO id={best_circle_id}", marker="o", ms=3, lw=1)
        plt.plot(tA, series["fix_opt"]["eta_no_sb"],  label=f"Fijo ÓPTIMA id={best_opt_id}", marker="o", ms=3, lw=1)
        plt.ylim(0,1.05); plt.grid(alpha=.3); plt.legend()
        plt.xlabel("Hora decimal [h]"); plt.ylabel("η (sin sombras/bloqueos)")
        plt.title("Comparativa — η(t) SIN sombras ni bloqueos")
        plt.tight_layout(); plt.savefig(exp / "comparativa_noSB_circle_optima.png", dpi=160); plt.show()

        # Heatmap con η_all media (círculo y óptima resaltados)
        all_ids = list(idx_xy.keys())
        XY_all = np.array([idx_xy[i] for i in all_ids], float)
        eta_media_all = np.array([np.mean(eta_all_acc.get(hid, [0])) for hid in all_ids])

        cnt_c = Counter(series["circle"]["ids"]); cnt_o = Counter(series["optima"]["ids"])
        ids_c, ids_o = list(cnt_c.keys()), list(cnt_o.keys())
        XY_c = np.array([idx_xy[i] for i in ids_c], float) if ids_c else np.zeros((0,2))
        XY_o = np.array([idx_xy[i] for i in ids_o], float) if ids_o else np.zeros((0,2))

        plt.figure(figsize=(8.8,7.6))
        sc = plt.scatter(XY_all[:,0], XY_all[:,1], c=eta_media_all, s=16, alpha=0.9)
        cb = plt.colorbar(sc); cb.set_label("η_all media (día)")
        if len(XY_h := XY_c):
            plt.scatter(XY_h[:,0], XY_h[:,1], s=60, marker="s", facecolors="none", edgecolors="limegreen", lw=1.6,
                        label=f"Usados círculo (n={len(ids_c)})")
        if len(XY_o):
            plt.scatter(XY_o[:,0], XY_o[:,1], s=60, marker="^", facecolors="none", edgecolors="orange", lw=1.6,
                        label=f"Usados óptima (n={len(ids_o)})")

        # Fijos
        xC, yC = idx_xy[best_circle_id]
        xO, yO = idx_xy[best_opt_id]
        plt.scatter([xC],[yC], s=140, marker="*", c="red", label=f"Fijo CÍRCULO id={best_circle_id}", zorder=5)
        plt.scatter([xO],[yO], s=140, marker="*", c="k",   label=f"Fijo ÓPTIMA id={best_opt_id}",  zorder=5)

        plt.axis("equal"); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("Campo — η_all media por heliostato (círculo + óptima + fijos)")
        plt.legend(loc="best"); plt.savefig(exp / "campo_heatmap_con_seleccionados_circle_optima.png", dpi=180); plt.show()

    # ===== MEDIAS (sin sombras/bloqueos) =====
    def mean_valid(v):
        vv = np.array(v, float); vv = vv[~np.isnan(vv)]
        return float(np.mean(vv)) if len(vv) else np.nan

    m_circ = mean_valid(series["circle"]["eta_no_sb"])
    m_opt  = mean_valid(series["optima"]["eta_no_sb"])
    m_fc   = mean_valid(series["fix_circle"]["eta_no_sb"])
    m_fo   = mean_valid(series["fix_opt"]["eta_no_sb"])

    print("\n==== MEDIAS η (SIN sombras ni bloqueos) ====")
    print(f"Círculo:      η̄_noSB = {m_circ:.4f}")
    print(f"Óptima:       η̄_noSB = {m_opt:.4f}")
    print(f"Fijo CÍRCULO: η̄_noSB = {m_fc:.4f}")
    print(f"Fijo ÓPTIMA:  η̄_noSB = {m_fo:.4f}")
    print("============================================")

    cp.data_free(r)

if __name__ == "__main__":
    main()
