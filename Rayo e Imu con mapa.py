# -*- coding: utf-8 -*-
"""
Helióstato integrado:
- Planificador solar en tiempo real (ψ = α_n − 90°, bisector, r óptimo con filtro y |dr/dt|).
- Percepción: Picamera2 + BNO055 → (x, y, ψ) en ENU con torre en (0,0,H).
- Control: feedforward + Pure-Pursuit/PD → v, w → vR, vL.
- ODrive (CANSimple): velocidad por rueda (rev/s) a 2 ejes.
- Actuador de inclinación por CAN: consigna de ángulo (grados) + monitor de ángulo real.

Requisitos:
  pip install opencv-python numpy python-can adafruit-circuitpython-bno055
  picamera2, board, busio (en Raspberry Pi)

Ajusta en Config:
  - lat/lon, rueda R, gear_ratio
  - IDs CAN (ODrive right/left, actuador)
  - signos de giro (sign_R, sign_L)
"""
from __future__ import annotations

import math, time, struct, threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

# ======================= CONFIGURACIÓN =======================
@dataclass
class Config:
    # Localización (Madrid ETSII por defecto)
    lat: float = 40.4396
    lon: float = -3.7274

    # Torre y radio
    tower_height: float = 3.0
    r_min: float = 2.0
    r_max: float = 15.0
    lambda_dist: float = 0.12

    # límites cinemáticos
    v_max: float = 0.6          # m/s
    w_max: float = 0.8          # rad/s
    dr_max: float = 0.6         # m/s
    r_alpha: float = 0.3        # filtro exp

    # base diferencial
    track_width: float = 0.50   # L (m)
    wheel_radius: float = 0.08  # R (m)
    gear_ratio: float = 1.0     # motor -> rueda

    # control
    lookahead: float = 0.8
    k_v: float = 0.6
    k_w: float = 1.4

    # lazo principal
    step_s: float = 0.5

    # --- CAN bus compartido ---
    can_channel: str = "can0"
    can_bitrate: int = 500_000

    # --- ODrive (CANSimple) ---
    node_right: int = 1      # ID nodo rueda derecha
    node_left: int  = 2      # ID nodo rueda izquierda
    vmax_rev_s_clip: float = 20.0  # clip de seguridad (rev/s)
    sign_R: float = +1.0     # invierte si tu rueda va al revés
    sign_L: float = +1.0

    # --- Actuador de inclinación (tu protocolo CAN) ---
    ACT_CAN_ID_CMD: int = 0x123  # envío comandos/ángulo
    ACT_CAN_ID_DATA: int = 0x124 # recibe objetivo, ángulo y error (h * 3)
    tilt_min_deg: float = -2.0
    tilt_max_deg: float = 95.0
    tilt_rate_limit_deg_s: float = 60.0  # límite software opcional


# ======================= UTILIDADES SOLARES =======================
def sunpos(y,m,d,hh,mm,ss,lat,lon):
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

def sun_dir_ENU(dt, lat, lon):
    A,z = sunpos(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second+dt.microsecond/1e6,lat,lon)
    sz = math.cos(z); sxy = math.sin(z)
    sx, sy, szU = math.sin(A)*sxy, math.cos(A)*sxy, sz
    return sx, sy, szU, A

def unit(v):
    n = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); 
    return (v[0]/n, v[1]/n, v[2]/n) if n>1e-9 else None

def azimuth_EN_from_vector(v):
    deg = math.degrees(math.atan2(v[0], v[1]))  # atan2(E,N)
    return (deg+360.0)%360.0

def wrap_angle(a):
    while a <= -math.pi: a += 2*math.pi
    while a >  math.pi:  a -= 2*math.pi
    return a

def position_from_sun(A_sun, radius):
    beta = (A_sun + math.pi)%(2*math.pi)
    east  = radius*math.sin(beta)
    north = radius*math.cos(beta)
    return (east, north, 0.0)

# ======================= PLANIFICADOR EN TIEMPO REAL =======================
@dataclass
class Command:
    t: datetime
    x: float; y: float; psi: float
    v: float; w: float
    r: float
    n_az_deg: float
    tilt_deg: float

def _efficiency_for_radius(r, A_sun, s, H, lam, r_max):
    pos = position_from_sun(A_sun, r)
    tx,ty,tz = -pos[0], -pos[1], H
    tdir = unit((tx,ty,tz))
    if not tdir: return 0.0
    n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2]))
    if not n: return 0.0
    cos_eff = max(0.0, min(1.0, s[0]*n[0]+s[1]*n[1]+s[2]*n[2]))
    penalty = lam * (r / r_max)**2
    return max(0.0, cos_eff - penalty)

def _optimize_radius(A_sun, s, cfg: Config):
    a, b = cfg.r_min, cfg.r_max
    if b <= a: return a
    gr = (math.sqrt(5.0)-1.0)/2.0
    c = b - gr*(b-a); d = a + gr*(b-a)
    fc = _efficiency_for_radius(c, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    fd = _efficiency_for_radius(d, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    for _ in range(28):
        if fc < fd:
            a = c; c = d; fc = fd
            d = a + gr*(b-a); fd = _efficiency_for_radius(d, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
        else:
            b = d; d = c; fd = fc
            c = b - gr*(b-a); fc = _efficiency_for_radius(c, A_sun, s, cfg.tower_height, cfg.lambda_dist, cfg.r_max)
    return (a+b)/2.0

class TrajectoryPlannerRT:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.r_prev = None; self.psi_prev = None
        self.x_prev = None; self.y_prev = None
        self.last_t = None

    def step(self, now_utc: datetime) -> Optional[Command]:
        dt_s = self.cfg.step_s if self.last_t is None else max(0.05, (now_utc - self.last_t).total_seconds())
        self.last_t = now_utc

        sx,sy,sz,A = sun_dir_ENU(now_utc, self.cfg.lat, self.cfg.lon)
        if sz <= 0.0:
            return None  # noche
        s = (sx,sy,sz)

        # radio óptimo + filtro
        r_opt = _optimize_radius(A, s, self.cfg)
        if self.r_prev is None:
            r_eff = r_opt
        else:
            r_filt = self.cfg.r_alpha*r_opt + (1.0-self.cfg.r_alpha)*self.r_prev
            dr_lim = self.cfg.dr_max*dt_s
            r_eff = max(self.cfg.r_min, min(self.cfg.r_max, self.r_prev + max(-dr_lim, min(dr_lim, r_filt - self.r_prev))))

        # posición y normal
        pos = position_from_sun(A, r_eff)
        tx,ty,tz = -pos[0], -pos[1], self.cfg.tower_height
        tdir = unit((tx,ty,tz))
        n = unit((s[0]+tdir[0], s[1]+tdir[1], s[2]+tdir[2])) if tdir else None
        if not n: return None

        n_az_rad = math.radians(azimuth_EN_from_vector(n))
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
        psi_d = wrap_angle(n_az_rad - math.pi/2)  # espejo a la derecha

        # feedforward (dif de muestra a muestra)
        if (self.psi_prev is None) or (self.x_prev is None):
            v_ff = 0.0; w_ff = 0.0
        else:
            dx = pos[0] - self.x_prev; dy = pos[1] - self.y_prev
            dist = math.hypot(dx, dy)
            v_ff = min(self.cfg.v_max, dist/dt_s)
            dpsi = wrap_angle(psi_d - self.psi_prev)
            w_ff = max(-self.cfg.w_max, min(self.cfg.w_max, dpsi/dt_s))
            if abs(w_ff) > 0.8*self.cfg.w_max: v_ff *= 0.6

        self.r_prev = r_eff; self.psi_prev = psi_d; self.x_prev, self.y_prev = pos[0], pos[1]

        return Command(now_utc, pos[0], pos[1], psi_d, v_ff, w_ff, r_eff, math.degrees(n_az_rad), tilt_deg)



class Visualizer:
    def __init__(self, map_size=450, scale_px_per_cm=4):
        import cv2
        self.cv2 = cv2
        self.map_size = map_size
        self.scale = scale_px_per_cm  # px por cm
        self._grid = self._make_grid()

    def _make_grid(self):
        ms = self.map_size
        grid = np.ones((ms, ms, 3), dtype=np.uint8) * 255
        c = (ms//2, ms//2)
        # cuadrícula
        grid_spacing = 90
        for i in range(c[0] % grid_spacing, ms, grid_spacing):
            self.cv2.line(grid, (i,0), (i,ms), (180,180,180), 1, self.cv2.LINE_AA)
            self.cv2.line(grid, (0,i), (ms,i), (180,180,180), 1, self.cv2.LINE_AA)
        # radial
        for r in range(60, ms//2, 90):
            self.cv2.circle(grid, c, r, (160,160,160), 1, self.cv2.LINE_AA)
        # ángulos cada 45°
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            x = int(c[0] + (ms//2.3)*math.cos(rad))
            y = int(c[1] - (ms//2.3)*math.sin(rad))
            self.cv2.line(grid, c, (x,y), (160,160,160), 1, self.cv2.LINE_AA)
        # marcas 0/90/180/270
        for ang in range(0, 360, 90):
            rad = math.radians(ang)
            x = int(c[0] + (ms//2.5)*math.cos(rad))
            y = int(c[1] - (ms//2.5)*math.sin(rad))
            self.cv2.putText(grid, f"{ang} º", (x,y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, self.cv2.LINE_AA)
        # ejes y torre
        self.cv2.line(grid, (0,c[1]), (ms,c[1]), (0,0,0), 2, self.cv2.LINE_AA)
        self.cv2.line(grid, (c[0],0), (c[0],ms), (0,0,0), 2, self.cv2.LINE_AA)
        self.cv2.circle(grid, c, 10, (0,0,255), -1)
        self.cv2.putText(grid, "Torre", (c[0]+15, c[1]-15), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, self.cv2.LINE_AA)
        return grid

    def _robot_px(self, X_cm, Y_cm):
        c = (self.map_size//2, self.map_size//2)
        rx = int(c[0] - X_cm * self.scale)   # Este→ +X (cm) mapea a izquierda en px (como tu script)
        ry = int(c[1] + Y_cm * self.scale)   # Norte→ +Y (cm) mapea a abajo en px
        return rx, ry

    def draw_map_in(self, frame_bgr, X_cm, Y_cm, yaw_err_deg):
        ms = self.map_size
        map_img = self._grid.copy()
        rx, ry = self._robot_px(X_cm, Y_cm)
        self.cv2.circle(map_img, (rx,ry), 10, (255,0,0), -1)
        self.cv2.putText(map_img, "ROBOT", (rx+15, ry-15), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, self.cv2.LINE_AA)
        L = 45
        ax = int(rx + L*math.sin(math.radians(yaw_err_deg)))
        ay = int(ry - L*math.cos(math.radians(yaw_err_deg)))
        self.cv2.arrowedLine(map_img, (rx,ry), (ax,ay), (255,255,0), 3, self.cv2.LINE_AA, tipLength=0.4)
        h, w, _ = frame_bgr.shape
        frame_bgr[h-ms:h, 0:ms] = map_img
        return frame_bgr

    def compose_band(self, w, distance_cm, distance_h_cm, offset_deg, yaw_act, yaw_err, X_cm, Y_cm, pitch_deg, bright_avg, bright_status, off_px):
        band_h = 150
        band = np.zeros((band_h, w, 3), dtype=np.uint8)
        cv2 = self.cv2
        t1 = f"Dist: {distance_cm:.1f}cm | Horiz: {distance_h_cm:.1f}cm | Offset: {offset_deg:.1f}°"
        t2 = f"Yaw act: {yaw_act:.2f}° | Error: {yaw_err:.2f}°"
        t3 = f"Pos: X={X_cm:.1f}cm, Y={Y_cm:.1f}cm"
        t4 = f"Pitch: {pitch_deg:.2f}°"
        t5 = f"Brightness: {bright_avg:.1f} | {bright_status}"
        if bright_status == "Not centered":
            t5 += f" (Off: {off_px:.1f}px)"
        cv2.putText(band, t1, (10,25),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(band, t2, (10,55),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(band, t3, (10,85),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(band, t4, (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(band, t5, (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        return band


# ======================= PERCEPCIÓN (CÁMARA + IMU) =======================

class Perception:
    """
    Devuelve:
      pose -> (x,y,psi) en ENU (m, rad) o None
      frame_u -> imagen rectificada
      marcas -> Nx2 (px)
      extras -> dict con: distance_cm, distance_h_cm, offset_deg, yaw_act, yaw_err, pitch_deg,
                          bright_avg, bright_status, off_px, X_cm, Y_cm
    """
    def __init__(self):
        import cv2, board, busio, adafruit_bno055
        from picamera2 import Picamera2
        self.cv2 = cv2
        # Calibración
        self.camera_matrix = np.array([[1.82798542e+03, 0.0, 5.72342464e+02],
                                       [0.0, 1.82504450e+03, 3.62602479e+02],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([[4.95174274e-03, 2.65701920e+00, -1.43227501e-03, -1.14715430e-02, -1.37391077e+01]], dtype=np.float32)
        # IMU
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=0x28)
        input("Coloca el robot mirando a la TORRE y pulsa ENTER…")
        self.yaw_ref_deg = self.sensor.euler[0] or 0.0
        # Cámara
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size   = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        time.sleep(0.2)
        # patrón
        self.REAL_WIDTH  = 6.0
        self.REAL_HEIGHT = 10.0
        self.object_points = np.array([[0,0,0],
                                       [self.REAL_WIDTH, 0, 0],
                                       [self.REAL_WIDTH, self.REAL_HEIGHT, 0],
                                       [0, self.REAL_HEIGHT, 0]], dtype=np.float32)
        self.FOV_HORIZONTAL = 62.2  # deg

    def _detectar_marcas(self, frame_u):
        cv2 = self.cv2
        gray = cv2.cvtColor(frame_u, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marcas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 1500:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"]/M["m00"]); cY = int(M["m01"]/M["m00"])
                    marcas.append([cX, cY])
        return np.array(marcas, dtype=np.float32)

    def _order4(self, pts):
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def get_pose(self):
        frame = self.picam2.capture_array()
        frame_u = self.cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        marcas = self._detectar_marcas(frame_u)

        # defaults
        distance_cm = 0.0
        distance_h_cm = 0.0
        bright_avg = 0.0
        bright_status = "No detection"
        off_px = 0.0
        yaw_act = self.sensor.euler[0] or 0.0
        pitch_deg = self.sensor.euler[2] or 0.0
        yaw_err = (yaw_act - self.yaw_ref_deg)
        yaw_err_rad = math.radians(yaw_err)

        if len(marcas) == 4:
            pts = self._order4(marcas).astype(np.float32)
            for (x,y) in pts:
                self.cv2.circle(frame_u, (int(x),int(y)), 5, (0,0,255), -1)

            ok, rvec, tvec = self.cv2.solvePnP(self.object_points, pts, self.camera_matrix, self.dist_coeffs, flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distance_cm = float(tvec[2][0])
            pitch_rad = math.radians(pitch_deg)
            distance_h_cm = distance_cm * max(0.0, math.cos(pitch_rad))  # ← corrección

            # centro + brillo
            cx, cy = np.mean(pts, axis=0).astype(int)
            gray = self.cv2.cvtColor(frame_u, self.cv2.COLOR_BGR2GRAY)
            win = 20
            x1 = max(cx - win//2, 0); x2 = min(cx + win//2, gray.shape[1]-1)
            y1 = max(cy - win//2, 0); y2 = min(cy + win//2, gray.shape[0]-1)
            avg = float(np.mean(gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else 0))
            bright_avg = avg
            thresh = 200
            if avg >= thresh:
                bright_status = "Correct"
                off_px = 0.0; bright_pt = (cx, cy)
            else:
                masked = np.zeros_like(gray); self.cv2.fillPoly(masked, [pts.astype(np.int32)], 255)
                masked = self.cv2.bitwise_and(gray, gray, mask=masked)
                _, _, _, maxLoc = self.cv2.minMaxLoc(masked)
                bright_pt = maxLoc
                off_px = float(np.linalg.norm(np.array(bright_pt) - np.array([cx,cy])))
            self.cv2.circle(frame_u, (cx,cy), 5, (255,0,0), 2)
            self.cv2.circle(frame_u, bright_pt, 5, (0,255,0), -1)
        else:
            h, w, _ = frame_u.shape
            cx, cy = int(w/2), int(h/2)

        # offset angular por píxeles
        h, w, _ = frame_u.shape
        if len(marcas) > 0: center = np.mean(marcas, axis=0)
        else: center = np.array([w/2, h/2])
        offset_x = float(center[0] - (w/2))
        offset_deg = (offset_x / (w/2)) * (self.FOV_HORIZONTAL/2.0)

        # posición ENU (cm) a partir de distancia horizontal + yaw_err
        X_cm = distance_h_cm * math.sin(yaw_err_rad)
        Y_cm = distance_h_cm * math.cos(yaw_err_rad)

        # Pose ENU en m (coherente con el controlador)
        x_m = X_cm/100.0; y_m = Y_cm/100.0
        psi = wrap_angle(math.radians(yaw_act))

        extras = dict(distance_cm=distance_cm, distance_h_cm=distance_h_cm,
                      offset_deg=offset_deg, yaw_act=yaw_act, yaw_err=yaw_err,
                      pitch_deg=pitch_deg, bright_avg=bright_avg,
                      bright_status=bright_status, off_px=off_px,
                      X_cm=X_cm, Y_cm=Y_cm, cx=int(center[0]), cy=int(center[1]))
        # si no hay 4 marcas, no devolvemos pose (para que el supervisor frene)
        pose = (x_m, y_m, psi) if len(marcas)==4 else None
        return pose, frame_u, marcas, extras


# ======================= CONTROL (FF + FB) =======================
class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    def compute(self, meas: Tuple[float,float,float], ref: Command):
        xm, ym, psim = meas
        dx = ref.x - xm; dy = ref.y - ym
        # error en marco robot
        ex_r =  math.cos(-psim)*dx - math.sin(-psim)*dy
        ey_r =  math.sin(-psim)*dx + math.cos(-psim)*dy
        # Pure-Pursuit
        Ld = self.cfg.lookahead
        kappa = 0.0
        if (ex_r**2 + ey_r**2) > 1e-6:
            scale = Ld / math.hypot(ex_r, ey_r)
            x_look = ex_r * scale; y_look = ey_r * scale
            kappa = (2.0 * y_look) / (Ld*Ld)
        e_psi = wrap_angle(ref.psi - psim)
        v_cmd = ref.v + self.cfg.k_v * ex_r
        w_cmd = ref.w + self.cfg.k_w * e_psi + v_cmd * kappa
        # límites
        v_cmd = max(-self.cfg.v_max, min(self.cfg.v_max, v_cmd))
        w_cmd = max(-self.cfg.w_max, min(self.cfg.w_max, w_cmd))
        return v_cmd, w_cmd

# ======================= CAN manager compartido =======================
class CanManager:
    def __init__(self, cfg: Config):
        import can
        self.can = can
        self.bus = can.Bus(interface="socketcan", channel=cfg.can_channel,
                           bitrate=cfg.can_bitrate, receive_own_messages=False)

    def shutdown(self):
        try:
            self.bus.shutdown()
        except Exception:
            pass

# ======================= ODrive por CANSimple =======================
SET_AXIS_STATE        = 0x07  # <u32>
GET_ENCODER_ESTIMATES = 0x09  # resp: <f32 pos_rev, f32 vel_rev_s>
SET_INPUT_VEL         = 0x0D  # <f32 vel_rev_s, f32 torque_ff>
AXIS_CLOSED_LOOP      = 8

class ODriveCanDriver:
    """
    Envía velocidad por CAN a dos ejes ODrive (rueda derecha/izquierda).
    vR/vL [m/s] → rev/s motor. Ajusta sign_R/sign_L en Config si el giro está invertido.
    """
    def __init__(self, cfg: Config, can_mgr: CanManager):
        self.cfg = cfg
        self.bus = can_mgr.bus
        self.enabled = False
        # Cerrar lazo en ambos ejes
        for nid in (cfg.node_right, cfg.node_left):
            msg_id = (nid << 5) | SET_AXIS_STATE
            self.bus.send(self._msg(msg_id, struct.pack("<I", AXIS_CLOSED_LOOP)))
            time.sleep(0.01)
        self.enabled = True

    def _msg(self, arb_id, data=b"", rtr=False, dlc=0):
        return self._mk_msg(arb_id, data, rtr, dlc)

    def _mk_msg(self, arb_id, data=b"", rtr=False, dlc=0):
        import can
        if rtr:
            return can.Message(arbitration_id=arb_id, is_extended_id=False, is_remote_frame=True, dlc=dlc)
        else:
            return can.Message(arbitration_id=arb_id, is_extended_id=False, data=data)

    def _ms_to_revs(self, v_ms: float) -> float:
        omega_wheel = v_ms / self.cfg.wheel_radius           # rad/s
        rev_s_motor = (omega_wheel / (2.0*math.pi)) * self.cfg.gear_ratio
        return rev_s_motor

    def send_wheel_speeds(self, vR_ms: float, vL_ms: float):
        if not self.enabled: return
        vR_rev = self.cfg.sign_R * self._ms_to_revs(vR_ms)
        vL_rev = self.cfg.sign_L * self._ms_to_revs(vL_ms)
        vmax = self.cfg.vmax_rev_s_clip
        vR_rev = max(-vmax, min(vmax, vR_rev))
        vL_rev = max(-vmax, min(vmax, vL_rev))
        # R
        msg_id_R = (self.cfg.node_right << 5) | SET_INPUT_VEL
        self.bus.send(self._msg(msg_id_R, struct.pack("<ff", float(vR_rev), 0.0)))
        # L
        msg_id_L = (self.cfg.node_left  << 5) | SET_INPUT_VEL
        self.bus.send(self._msg(msg_id_L, struct.pack("<ff", float(vL_rev), 0.0)))

    def stop(self):
        try:
            self.send_wheel_speeds(0.0, 0.0)
        except Exception:
            pass

# ======================= Actuador de inclinación por CAN (usuario) =======================
class TiltActuator:
    """
    Envía consigna de inclinación (grados) al Arduino por CAN.
    - CAN_ID_CMD: dos bytes little-endian con el ángulo en grados (int, como tu enviar_angulo()).
    - CAN_ID_DATA: tres enteros h (objetivo, ángulo, error) escalados /100 para telemetría (como tu monitor()).
    - Aplica límites mecánicos y rate limit opcional.
    """
    def __init__(self, cfg: Config, can_mgr: CanManager):
        self.cfg = cfg
        self.bus = can_mgr.bus
        self.last_cmd_deg = None
        self.last_ts = None
        self.telemetry = {"obj": None, "ang": None, "err": None}
        self._run_monitor = True
        self._th = threading.Thread(target=self._monitor_thread, daemon=True)
        self._th.start()

    def set_tilt_deg(self, tilt_deg: float):
        # límites mecánicos
        t = max(self.cfg.tilt_min_deg, min(self.cfg.tilt_max_deg, float(tilt_deg)))

        # rate limit opcional
        now = time.perf_counter()
        if self.last_cmd_deg is not None and self.last_ts is not None:
            dt = max(1e-3, now - self.last_ts)
            max_step = self.cfg.tilt_rate_limit_deg_s * dt
            if abs(t - self.last_cmd_deg) > max_step:
                t = self.last_cmd_deg + math.copysign(max_step, t - self.last_cmd_deg)

        # empaquetar dos bytes (entero grados). Mantengo tu protocolo (sin *100).
        val = int(round(t))
        data = struct.pack("<h", val)  # signed 16-bit
        try:
            import can
            msg = can.Message(arbitration_id=self.cfg.ACT_CAN_ID_CMD, data=data, is_extended_id=False)
            self.bus.send(msg)
        except Exception as e:
            print("❌ Error CAN tilt:", e)

        self.last_cmd_deg = t
        self.last_ts = now

    def _monitor_thread(self):
        """Lee CAN_ID_DATA y actualiza telemetría (objetivo, angulo, error) en °."""
        import can
        print("👂 Monitor de tilt por CAN…")
        while self._run_monitor:
            try:
                msg = self.bus.recv(0.2)
                if not msg or msg.arbitration_id != self.cfg.ACT_CAN_ID_DATA or len(msg.data) < 6:
                    continue
                objetivo, angulo, error = struct.unpack("<hhh", msg.data[:6])
                self.telemetry["obj"] = objetivo / 100.0
                self.telemetry["ang"] = angulo   / 100.0
                self.telemetry["err"] = error    / 100.0
                # (opcional) imprime de vez en cuando
                # print(f"🎯 {self.telemetry['obj']:7.2f}° | 📐 {self.telemetry['ang']:7.2f}° | ❌ {self.telemetry['err']:7.2f}°")
            except Exception:
                pass

    def stop(self):
        self._run_monitor = False

# ======================= SUPERVISOR =======================
class Supervisor:
    def __init__(self, cfg=Config()):
        self.cfg  = cfg
        self.plan = TrajectoryPlannerRT(cfg)
        self.perc = Perception()
        self.ctrl = Tracker(cfg)
        self.canm = CanManager(cfg)
        self.drive= ODriveCanDriver(cfg, self.canm)
        self.tilt = TiltActuator(cfg, self.canm)
        self.viz  = Visualizer(map_size=450, scale_px_per_cm=4)   # ← tu mapa
        self.state = "INIT"

    def step(self, now_utc: datetime):
        try:
            if self.state == "INIT":
                self.state = "TRACK"

            if self.state == "TRACK":
                ref = self.plan.step(now_utc)
                pose, frame_u, marcas, ex = self.perc.get_pose()

                if ref is None:
                    self.drive.send_wheel_speeds(0.0, 0.0)
                    # Visual solo
                    frame_map = self.viz.draw_map_in(frame_u.copy(), ex["X_cm"], ex["Y_cm"], ex["yaw_err"])
                    band = self.viz.compose_band(frame_u.shape[1],
                                                 ex["distance_cm"], ex["distance_h_cm"], ex["offset_deg"],
                                                 ex["yaw_act"], ex["yaw_err"], ex["X_cm"], ex["Y_cm"],
                                                 ex["pitch_deg"], ex["bright_avg"], ex["bright_status"], ex["off_px"])
                    final = np.vstack([frame_map, band])
                    import cv2; cv2.imshow("Visualizacion", final); cv2.waitKey(1)
                    return

                # ACTUADOR tilt
                self.tilt.set_tilt_deg(ref.tilt_deg)

                if pose is None:
                    self.drive.send_wheel_speeds(0.0, 0.0)
                else:
                    v_cmd, w_cmd = self.ctrl.compute(pose, ref)
                    L = self.cfg.track_width
                    vR = v_cmd + 0.5*L*w_cmd
                    vL = v_cmd - 0.5*L*w_cmd
                    vmax_wheel = self.cfg.v_max + 0.5*L*self.cfg.w_max
                    scale = max(1.0, max(abs(vR), abs(vL))/vmax_wheel)
                    vR /= scale; vL /= scale
                    self.drive.send_wheel_speeds(vR, vL)

                # --- Interfaz (siempre) ---
                frame_map = self.viz.draw_map_in(frame_u.copy(), ex["X_cm"], ex["Y_cm"], ex["yaw_err"])
                band = self.viz.compose_band(frame_u.shape[1],
                                             ex["distance_cm"], ex["distance_h_cm"], ex["offset_deg"],
                                             ex["yaw_act"], ex["yaw_err"], ex["X_cm"], ex["Y_cm"],
                                             ex["pitch_deg"], ex["bright_avg"], ex["bright_status"], ex["off_px"])
                final = np.vstack([frame_map, band])
                import cv2; cv2.imshow("Visualizacion", final); cv2.waitKey(1)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("FAULT:", e)
            self.state = "FAULT"
            self.drive.stop()

    def shutdown(self):
        try: self.drive.stop()
        except: pass
        try: self.tilt.stop()
        except: pass
        try: self.canm.shutdown()
        except: pass


# ======================= MAIN =======================
def main():
    cfg = Config()
    sup = Supervisor(cfg)
    print("Helióstato: TRACK (feedforward solar + feedback cámara/IMU + ODrive + Tilt CAN). Ctrl-C para salir.")
    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            sup.step(now_utc)
            time.sleep(cfg.step_s)
    except KeyboardInterrupt:
        print("\nApagando…")
    finally:
        sup.shutdown()

if __name__ == "__main__":
    main()
