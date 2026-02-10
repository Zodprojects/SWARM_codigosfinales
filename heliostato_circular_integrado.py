# -*- coding: utf-8 -*-
"""
HELIÓSTATO CIRCULAR INTEGRADO CON FEEDBACK COMPLETO + CORRECCIÓN AUTOMÁTICA DE RAYO

Integra:
- Planificador de trayectoria CIRCULAR con TILT (de mi_software_circunferencia_con_tilt.py)
- ODrive + 2 ruedas (control diferencial)
- Actuador lineal CAN (tilt) con "Control con feedback definitivo"
- 2 IMUs: BNO055 (orientación global) + MPU6050 (control tilt - ROLL)
- Cámara Picamera2 (feedback posición visual + detección de rayo)
- **CORRECCIÓN AUTOMÁTICA**: 
  - Rayo arriba/abajo → ajusta TILT del espejo
  - Rayo izquierda/derecha → ajusta POSICIÓN del robot

Requisitos:
  pip install opencv-python numpy python-can adafruit-circuitpython-bno055 smbus2
  picamera2, board, busio (en Raspberry Pi)

Configurar can0:
  sudo ip link set can0 up type can bitrate 500000
"""

from __future__ import annotations
import math, time, struct, threading, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

# ======================= CONFIGURACIÓN =======================
@dataclass
class Config:
    # Localización (Madrid ETSII por defecto - ajusta a tu ubicación)
    lat: float = 40.4396
    lon: float = -3.7274

    # Torre y radio CIRCULAR
    tower_height: float = 3.0
    r_circular: float = 8.0          # Radio fijo de la circunferencia (ajusta según tu campo)
    lambda_dist: float = 0.12        # Factor de eficiencia

    # Límites cinemáticos
    v_max: float = 0.6          # m/s
    w_max: float = 0.8          # rad/s

    # Base diferencial
    track_width: float = 0.50   # L (m) - separación entre ruedas
    wheel_radius: float = 0.08  # R (m) - radio de rueda
    gear_ratio: float = 1.0     # motor -> rueda

    # Control feedback
    lookahead: float = 0.8      # Pure-Pursuit lookahead
    k_v: float = 0.6            # Ganancia velocidad
    k_w: float = 1.4            # Ganancia angular

    # Lazo principal
    step_s: float = 0.5         # Período del supervisor (Hz)

    # --- CAN bus compartido ---
    can_channel: str = "can0"
    can_bitrate: int = 500_000

    # --- ODrive (CANSimple) ---
    node_right: int = 1                # ID nodo rueda derecha
    node_left: int = 2                 # ID nodo rueda izquierda
    vmax_rev_s_clip: float = 20.0     # Clip de seguridad (rev/s)
    sign_R: float = +1.0               # Invierte si tu rueda va al revés
    sign_L: float = +1.0

    # --- Actuador linear tilt (dir/PWM por CAN) ---
    ACT_CAN_ID_CMD: int = 0x321        # ID comando tilt (dir/PWM)
    ACT_CAN_ID_DATA: int = 0x322       # ID telemetría (opcional)
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 90.0
    
    # Control PID del actuador (configuración de "Control con feedback definitivo")
    Kp: float = 1.0                    # Ganancia proporcional
    Ki: float = 0.0                    # Ganancia integral
    Kd: float = 0.0                    # Ganancia derivada
    I_clamp: float = 0.6               # Límite integral anti-windup
    deadband_deg: float = 1.0          # Deadband (grados)
    PWM_MAX: int = 255                 # PWM máximo
    PWM_MIN_EFF: int = 85              # PWM mínimo efectivo
    PWM_SLEW: int = 25                 # Slew rate (cambio máximo por ciclo)
    pwm_per_degree: float = 12.0       # PWM adicional por grado de error
    can_send_interval: float = 0.1     # Intervalo mínimo entre envíos CAN (100ms)
    stall_timeout_s: float = 3.0       # Timeout para detección de estancamiento
    telemetry_hz: float = 5.0          # Frecuencia de impresión

    # --- IMU MPU6050 (validación tilt) ---
    mpu_bus: int = 1
    mpu_addr: int = 0x68
    mpu_dt: float = 0.02
    mpu_alpha: float = 0.98

# ======================= UTILIDADES SOLARES =======================
Vec3 = Tuple[float, float, float]

def sunpos(y, m, d, hh, mm, ss, lat, lon):
    """Posición solar (azimut, elevación) en grados."""
    from math import sin, cos, tan, asin, acos, atan2, radians, degrees
    d0 = 367*y - int(7*(y + int((m+9)/12))/4) + int(275*m/9) + d - 730531.5
    w = 282.9400 + 4.70935e-5*d0; e = 0.016709 - 1.151e-9*d0; M = (356.0470 + 0.9856002585*d0) % 360
    L = (w + M) % 360; E = M
    for _ in range(10):
        E = M + degrees(e*sin(radians(E)))
    x_ecl = cos(radians(E)) - e; y_ecl = sin(radians(E))*((1-e**2)**0.5); r = (x_ecl**2 + y_ecl**2)**0.5
    v = degrees(atan2(y_ecl, x_ecl)); lon_sun = (v + w) % 360
    x_eq = r*cos(radians(lon_sun)); y_eq = r*sin(radians(lon_sun))*cos(radians(23.44)); z_eq = r*sin(radians(lon_sun))*sin(radians(23.44))
    RA = degrees(atan2(y_eq, x_eq)) % 360; dec = degrees(asin(z_eq/r))
    UT = hh + mm/60.0 + ss/3600.0; LST = (100.46 + 0.985647*d0 + lon + 15*UT) % 360; HA = (LST - RA + 360) % 360
    if HA > 180: HA -= 360
    lat_r = radians(lat); dec_r = radians(dec); HA_r = radians(HA)
    alt = degrees(asin(sin(lat_r)*sin(dec_r) + cos(lat_r)*cos(dec_r)*cos(HA_r)))
    az = degrees(atan2(-cos(dec_r)*sin(HA_r), sin(dec_r)*cos(lat_r) - cos(dec_r)*sin(lat_r)*cos(HA_r)))
    return (az % 360, alt)

def sun_dir_ENU(dt: datetime, lat: float, lon: float):
    """Vector solar en ENU + azimut"""
    az_deg, alt_deg = sunpos(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, lat, lon)
    az_rad = math.radians(az_deg); alt_rad = math.radians(alt_deg)
    sx = math.cos(alt_rad)*math.sin(az_rad)
    sy = math.cos(alt_rad)*math.cos(az_rad)
    sz = math.sin(alt_rad)
    return sx, sy, sz, az_rad

def unit(v: Vec3) -> Vec3:
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0]/mag, v[1]/mag, v[2]/mag) if mag > 1e-9 else (0,0,1)

def azimuth_EN_from_vector(v: Vec3) -> float:
    """Azimut ENU en grados [0,360)"""
    return (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0

def wrap_angle(a):
    """Normaliza ángulo a [-π, π]"""
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

def position_from_sun(A_sun: float, radius: float):
    """Posición ENU en circunferencia opuesta al sol"""
    beta = (A_sun + math.pi) % (2*math.pi)
    east = radius * math.sin(beta)
    north = radius * math.cos(beta)
    return (east, north, 0.0)

# ======================= COMANDO =======================
@dataclass
class Command:
    t: datetime
    x: float
    y: float
    psi: float       # orientación deseada (rad)
    v: float         # velocidad lineal feedforward (m/s)
    w: float         # velocidad angular feedforward (rad/s)
    r: float         # radio circular
    n_az_deg: float  # azimut normal espejo (°)
    tilt_deg: float  # inclinación espejo (°)

# ======================= PLANIFICADOR CIRCULAR =======================
class CircularTrajectoryPlanner:
    """Planifica trayectoria CIRCULAR con tilt basada en posición solar"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_t = None
        self.psi_prev = None
        self.x_prev = None
        self.y_prev = None

    def step(self, now_utc: datetime) -> Optional[Command]:
        dt_s = self.cfg.step_s if self.last_t is None else max(0.05, (now_utc - self.last_t).total_seconds())
        self.last_t = now_utc

        # Dirección solar
        sx, sy, sz, A = sun_dir_ENU(now_utc, self.cfg.lat, self.cfg.lon)
        if sz <= 0.0:
            return None  # Noche - no hay sol
        s = (sx, sy, sz)

        # Posición en circunferencia (radio fijo)
        r_circ = self.cfg.r_circular
        pos = position_from_sun(A, r_circ)

        # Vector torre -> dispositivo
        tx, ty, tz = -pos[0], -pos[1], self.cfg.tower_height
        tdir = unit((tx, ty, tz))

        # Normal del espejo (bisector sol-torre)
        n = unit((s[0] + tdir[0], s[1] + tdir[1], s[2] + tdir[2]))

        # Azimut de la normal y tilt
        n_az_rad = math.radians(azimuth_EN_from_vector(n))
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))

        # Orientación del dispositivo (normal - 90°, espejo a la derecha)
        psi_d = wrap_angle(n_az_rad - math.pi/2)

        # Feedforward de velocidades (diferencia entre muestras)
        if self.psi_prev is None or self.x_prev is None:
            v_ff = 0.0
            w_ff = 0.0
        else:
            dx = pos[0] - self.x_prev
            dy = pos[1] - self.y_prev
            dist = math.hypot(dx, dy)
            v_ff = min(self.cfg.v_max, dist / dt_s)

            dpsi = wrap_angle(psi_d - self.psi_prev)
            w_ff = max(-self.cfg.w_max, min(self.cfg.w_max, dpsi / dt_s))

            # Reduce velocidad si gira mucho
            if abs(w_ff) > 0.8 * self.cfg.w_max:
                v_ff *= 0.6

        # Actualizar estado anterior
        self.psi_prev = psi_d
        self.x_prev, self.y_prev = pos[0], pos[1]

        return Command(now_utc, pos[0], pos[1], psi_d, v_ff, w_ff, r_circ, math.degrees(n_az_rad), tilt_deg)

# ======================= PERCEPCIÓN (CÁMARA + IMUs) =======================
class Perception:
    """Cámara Picamera2 + BNO055 (orientación) + MPU6050 (validación tilt)"""
    def __init__(self, cfg: Config):
        import cv2, board, busio, adafruit_bno055
        from picamera2 import Picamera2
        from smbus2 import SMBus

        self.cv2 = cv2
        self.cfg = cfg

        # Calibración cámara (ajusta con tus valores)
        self.camera_matrix = np.array([[1.82798542e+03, 0.0, 5.72342464e+02],
                                       [0.0, 1.82504450e+03, 3.62602479e+02],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([[4.95174274e-03, 2.65701920e+00, -1.43227501e-03,
                                     -1.14715430e-02, -1.37391077e+01]], dtype=np.float32)

        # BNO055 (orientación global)
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=0x28)
        input("📍 Coloca el robot mirando a la TORRE y pulsa ENTER...")
        self.yaw_ref_deg = self.sensor.euler[0] or 0.0

        # Cámara
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        time.sleep(0.2)

        # Patrón objetivo (4 marcas blancas)
        self.REAL_WIDTH = 6.0   # cm
        self.REAL_HEIGHT = 10.0  # cm
        self.object_points = np.array([[0, 0, 0],
                                       [self.REAL_WIDTH, 0, 0],
                                       [self.REAL_WIDTH, self.REAL_HEIGHT, 0],
                                       [0, self.REAL_HEIGHT, 0]], dtype=np.float32)
        self.FOV_HORIZONTAL = 62.2  # grados

    def _detectar_marcas(self, frame_u):
        gray = self.cv2.cvtColor(frame_u, self.cv2.COLOR_BGR2GRAY)
        blur = self.cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = self.cv2.threshold(blur, 180, 255, self.cv2.THRESH_BINARY)
        contours, _ = self.cv2.findContours(thresh, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        marcas = []
        for cnt in contours:
            area = self.cv2.contourArea(cnt)
            if 30 < area < 1500:
                M = self.cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    marcas.append([cX, cY])
        return np.array(marcas, dtype=np.float32)

    def _order4(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_pose(self):
        """Retorna (x, y, psi) en ENU [m, rad] + extras con detección del rayo"""
        frame = self.picam2.capture_array()
        frame_u = self.cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        marcas = self._detectar_marcas(frame_u)

        # Defaults
        distance_cm = 0.0
        distance_h_cm = 0.0
        bright_avg = 0.0
        bright_status = "No detection"
        off_px = 0.0
        offset_deg = 0.0
        yaw_act = self.sensor.euler[0] or 0.0
        pitch_deg = self.sensor.euler[2] or 0.0
        yaw_err = yaw_act - self.yaw_ref_deg
        yaw_err_rad = math.radians(yaw_err)

        if len(marcas) == 4:
            pts = self._order4(marcas).astype(np.float32)
            
            # Dibujar marcas detectadas
            for (x,y) in pts:
                self.cv2.circle(frame_u, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # solvePnP para distancia
            ok, rvec, tvec = self.cv2.solvePnP(self.object_points, pts, self.camera_matrix,
                                               self.dist_coeffs, flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distance_cm = float(tvec[2][0])
            pitch_rad = math.radians(pitch_deg)
            distance_h_cm = distance_cm * max(0.0, math.cos(pitch_rad))

            # Centro de las 4 marcas
            cx, cy = np.mean(pts, axis=0).astype(int)

            # **DETECCIÓN DEL RAYO** (punto de máxima luminosidad)
            gray = self.cv2.cvtColor(frame_u, self.cv2.COLOR_BGR2GRAY)
            
            # Ventana pequeña centrada en el centro de las marcas (20x20 px)
            win = 20
            x1 = max(cx - win//2, 0)
            x2 = min(cx + win//2, gray.shape[1]-1)
            y1 = max(cy - win//2, 0)
            y2 = min(cy + win//2, gray.shape[0]-1)
            
            # Brillo promedio en el centro
            center_window = gray[y1:y2, x1:x2] if (y2>y1 and x2>x1) else gray[cy:cy+1, cx:cx+1]
            bright_avg = float(np.mean(center_window))
            
            # Umbral de brillo para determinar si el rayo está centrado
            thresh = 200
            if bright_avg >= thresh:
                # Rayo CENTRADO en las marcas ✅
                bright_status = "Correct"
                off_px = 0.0
                bright_pt = (cx, cy)
            else:
                # Rayo DESCENTRADO - buscar punto más brillante dentro del cuadrilátero ⚠️
                bright_status = "Not centered"
                
                # Crear máscara del cuadrilátero
                masked = np.zeros_like(gray)
                self.cv2.fillPoly(masked, [pts.astype(np.int32)], 255)
                masked_gray = self.cv2.bitwise_and(gray, gray, mask=masked)
                
                # Encontrar punto de máxima luminosidad
                _, _, _, maxLoc = self.cv2.minMaxLoc(masked_gray)
                bright_pt = maxLoc
                off_px = float(np.linalg.norm(np.array(bright_pt) - np.array([cx,cy])))
            
            # Dibujar en frame: centro (azul) y punto brillante (verde)
            self.cv2.circle(frame_u, (cx, cy), 5, (255, 0, 0), 2)  # Azul = centro marcas
            self.cv2.circle(frame_u, bright_pt, 5, (0, 255, 0), -1)  # Verde = rayo

            # Offset angular (basado en el centro de las marcas)
            h, w, _ = frame_u.shape
            center = np.mean(pts, axis=0)
            offset_x = float(center[0] - (w/2))
            offset_deg = (offset_x / (w/2)) * (self.FOV_HORIZONTAL/2.0)
        else:
            # Sin marcas - usar centro de imagen
            h, w, _ = frame_u.shape
            cx, cy = int(w/2), int(h/2)
            bright_pt = (cx, cy)  # Por defecto en el centro

        # Posición ENU (cm)
        X_cm = distance_h_cm * math.sin(yaw_err_rad)
        Y_cm = distance_h_cm * math.cos(yaw_err_rad)

        # Posición ENU (m)
        x_m = X_cm / 100.0
        y_m = Y_cm / 100.0
        psi = wrap_angle(math.radians(yaw_act))

        # Extras para telemetría (incluye estado del rayo y coordenadas para corrección)
        extras = {
            "distance_cm": distance_cm,
            "distance_h_cm": distance_h_cm,
            "offset_deg": offset_deg,
            "yaw_act": yaw_act,
            "yaw_err": yaw_err,
            "pitch_deg": pitch_deg,
            "X_cm": X_cm,
            "Y_cm": Y_cm,
            "bright_avg": bright_avg,
            "bright_status": bright_status,
            "off_px": off_px,
            "bright_pt_x": bright_pt[0],  # Coordenada X del rayo (para corrección horizontal)
            "bright_pt_y": bright_pt[1],  # Coordenada Y del rayo (para corrección vertical)
            "center_x": cx,               # Centro X de las marcas
            "center_y": cy                # Centro Y de las marcas
        }

        # Si no hay 4 marcas, no devuelve pose (supervisor frena)
        pose = (x_m, y_m, psi) if len(marcas) == 4 else None
        return pose, frame_u, marcas, extras

# ======================= MPU6050 (Validación Tilt) =======================
class MPU6050:
    """IMU MPU6050 para validar tilt del espejo"""
    def __init__(self, cfg: Config):
        from smbus2 import SMBus
        self.bus = SMBus(cfg.mpu_bus)
        self.addr = cfg.mpu_addr
        self.dt = cfg.mpu_dt
        self.alpha = cfg.mpu_alpha

        PWR_MGMT_1 = 0x6B
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)
        time.sleep(0.05)

        self.gyro_bias = self._calibrate_gyro()
        ax, ay, az, *_ = self._read_accel_gyro()
        self.pitch, self.roll = self._accel_to_angles(ax, ay, az)

    def _read_word(self, reg):
        hi = self.bus.read_byte_data(self.addr, reg)
        lo = self.bus.read_byte_data(self.addr, reg + 1)
        val = (hi << 8) | lo
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def _read_accel_gyro(self):
        ACCEL_XOUT_H = 0x3B
        GYRO_XOUT_H = 0x43
        ACCEL_SCALE = 16384.0
        GYRO_SCALE = 131.0

        ax = self._read_word(ACCEL_XOUT_H) / ACCEL_SCALE
        ay = self._read_word(ACCEL_XOUT_H + 2) / ACCEL_SCALE
        az = self._read_word(ACCEL_XOUT_H + 4) / ACCEL_SCALE
        gx = self._read_word(GYRO_XOUT_H) / GYRO_SCALE
        gy = self._read_word(GYRO_XOUT_H + 2) / GYRO_SCALE
        gz = self._read_word(GYRO_XOUT_H + 4) / GYRO_SCALE
        return ax, ay, az, gx, gy, gz

    def _accel_to_angles(self, ax, ay, az):
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return pitch, roll

    def _calibrate_gyro(self, samples=200):
        sx = sy = sz = 0.0
        for _ in range(samples):
            _, _, _, gx, gy, gz = self._read_accel_gyro()
            sx += gx
            sy += gy
            sz += gz
            time.sleep(0.002)
        return {'x': sx / samples, 'y': sy / samples, 'z': sz / samples}

    def read_pitch_roll(self):
        ax, ay, az, gx, gy, gz = self._read_accel_gyro()
        gx -= self.gyro_bias['x']
        gy -= self.gyro_bias['y']
        pitch_gyro = self.pitch + gy * self.dt
        roll_gyro = self.roll + gx * self.dt
        pitch_acc, roll_acc = self._accel_to_angles(ax, ay, az)
        self.pitch = self.alpha * pitch_gyro + (1 - self.alpha) * pitch_acc
        self.roll = self.alpha * roll_gyro + (1 - self.alpha) * roll_acc
        return self.pitch, self.roll

# ======================= CONTROL (FEEDBACK) =======================
class Tracker:
    """Control Pure-Pursuit + PD para seguimiento de trayectoria"""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(self, meas: Tuple[float, float, float], ref: Command):
        xm, ym, psim = meas
        dx = ref.x - xm
        dy = ref.y - ym

        # Error en marco robot
        ex_r = math.cos(-psim) * dx - math.sin(-psim) * dy
        ey_r = math.sin(-psim) * dx + math.cos(-psim) * dy

        # Pure-Pursuit
        Ld = self.cfg.lookahead
        kappa = 0.0
        if (ex_r**2 + ey_r**2) > 1e-6:
            scale = Ld / math.hypot(ex_r, ey_r)
            x_look = ex_r * scale
            y_look = ey_r * scale
            kappa = (2.0 * y_look) / (Ld * Ld)

        e_psi = wrap_angle(ref.psi - psim)
        v_cmd = ref.v + self.cfg.k_v * ex_r
        w_cmd = ref.w + self.cfg.k_w * e_psi + v_cmd * kappa

        # Límites
        v_cmd = max(-self.cfg.v_max, min(self.cfg.v_max, v_cmd))
        w_cmd = max(-self.cfg.w_max, min(self.cfg.w_max, w_cmd))

        return v_cmd, w_cmd

# ======================= CAN MANAGER =======================
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

# ======================= ODRIVE (2 RUEDAS) =======================
SET_AXIS_STATE = 0x07
SET_INPUT_VEL = 0x0D
AXIS_CLOSED_LOOP = 8

class ODriveCanDriver:
    """Control de velocidad por rueda (vR, vL) via CAN"""
    def __init__(self, cfg: Config, can_mgr: CanManager):
        self.cfg = cfg
        self.bus = can_mgr.bus
        self.enabled = False

        # Cerrar lazo en ambos ejes
        for nid in (cfg.node_right, cfg.node_left):
            msg_id = (nid << 5) | SET_AXIS_STATE
            self.bus.send(self._mk_msg(msg_id, struct.pack("<I", AXIS_CLOSED_LOOP)))
            time.sleep(0.01)
        self.enabled = True

    def _mk_msg(self, arb_id, data=b""):
        import can
        return can.Message(arbitration_id=arb_id, is_extended_id=False, data=data)

    def _ms_to_revs(self, v_ms: float) -> float:
        omega_wheel = v_ms / self.cfg.wheel_radius  # rad/s
        rev_s_motor = (omega_wheel / (2.0 * math.pi)) * self.cfg.gear_ratio
        return rev_s_motor

    def send_wheel_speeds(self, vR_ms: float, vL_ms: float):
        if not self.enabled:
            return

        vR_rev = self.cfg.sign_R * self._ms_to_revs(vR_ms)
        vL_rev = self.cfg.sign_L * self._ms_to_revs(vL_ms)

        vmax = self.cfg.vmax_rev_s_clip
        vR_rev = max(-vmax, min(vmax, vR_rev))
        vL_rev = max(-vmax, min(vmax, vL_rev))

        # Rueda derecha
        msg_id_R = (self.cfg.node_right << 5) | SET_INPUT_VEL
        self.bus.send(self._mk_msg(msg_id_R, struct.pack("<ff", float(vR_rev), 0.0)))

        # Rueda izquierda
        msg_id_L = (self.cfg.node_left << 5) | SET_INPUT_VEL
        self.bus.send(self._mk_msg(msg_id_L, struct.pack("<ff", float(vL_rev), 0.0)))

    def stop(self):
        try:
            self.send_wheel_speeds(0.0, 0.0)
        except Exception:
            pass

# ======================= ACTUADOR TILT (dir/PWM + PID con MPU6050) =======================
# --- Direcciones del actuador (Control con feedback definitivo) ---
DIR_EXTIENDE = 0  # Aumenta ángulo (roll)
DIR_RETRAE = 1    # Disminuye ángulo (roll)

class TiltActuator:
    """
    Control PID del actuador de tilt usando MPU6050 como feedback.
    Implementación de "Control con feedback definitivo":
    - Usa ROLL exclusivamente (no pitch)
    - DIR_EXTIENDE=0, DIR_RETRAE=1
    - Intervalo mínimo entre envíos CAN
    - Telemetría específica
    """
    def __init__(self, cfg: Config, can_mgr: CanManager, mpu: 'MPU6050'):
        self.cfg = cfg
        self.bus = can_mgr.bus
        self.mpu = mpu
        self._target_deg = None
        self._dt = cfg.mpu_dt
        
        # Estado PID
        self._ei = 0.0
        self._e_prev = 0.0
        self._pwm_prev = 0
        
        # Control de envío CAN
        self._last_can_send = 0.0
        
        # Detección de estancamiento y telemetría
        self._last_theta = None
        self._last_change_t = time.perf_counter()
        self._last_telemetry = 0.0

    def set_tilt_deg(self, tilt_deg: float):
        """Establece el setpoint de tilt en grados"""
        self._target_deg = max(self.cfg.tilt_min_deg, 
                              min(self.cfg.tilt_max_deg, float(tilt_deg)))

    def update(self):
        """Ejecuta un ciclo de control PID (Control con feedback definitivo)"""
        if self._target_deg is None:
            return

        t0 = time.perf_counter()
        
        # Leer ángulos del MPU6050
        pitch, roll = self.mpu.read_pitch_roll()
        if pitch is None or roll is None:
            print("\n⚠️ IMU no válida")
            self._send(0, 0)
            return

        # **USA SOLO ROLL** (clave de "Control con feedback definitivo")
        theta_med = roll * 1.0  # cfg.imu_sign si necesitas invertir

        # Límites de seguridad
        if theta_med < self.cfg.tilt_min_deg or theta_med > self.cfg.tilt_max_deg:
            print(f"\n⛔ Ángulo fuera de rango: {theta_med:.1f}°. Paro.")
            self._send(0, 0)
            return

        # Cálculo del error y acción de control
        e = self._target_deg - theta_med
        u = self._pid(e, self._dt)
        
        # Conversión a PWM y dirección
        pwm = self._pwm_from_u(u)
        
        if pwm == 0:
            self._send(0, 0)
            direccion = 0
        else:
            direccion = self._dir_from_u(u)
            
            # **Solo envía CAN si ha pasado el intervalo mínimo**
            now = time.perf_counter()
            if now - self._last_can_send >= self.cfg.can_send_interval:
                try:
                    self._send(direccion, pwm)
                    self._last_can_send = now
                except Exception as e_can:
                    print(f"\n⚠️ Error CAN: {e_can}. Reintentando...")
                    time.sleep(0.05)  # Pausa antes de reintentar
        
        # Telemetría (formato específico de "Control con feedback definitivo")
        now = time.perf_counter()
        if now - self._last_telemetry >= (1.0 / max(1.0, self.cfg.telemetry_hz)):
            self._print_status(self._target_deg, theta_med, e, pwm, direccion)
            self._last_telemetry = now

    def _pid(self, e, dt):
        """Calcula la acción PID: u = Kp*e + Ki*ei + Kd*de"""
        # Integral con anti-windup
        self._ei += e * dt
        self._ei = max(-self.cfg.I_clamp, min(self.cfg.I_clamp, self._ei))
        
        # Derivada
        de = (e - self._e_prev) / max(1e-3, dt)
        self._e_prev = e
        
        # Salida PID
        u = self.cfg.Kp * e + self.cfg.Ki * self._ei + self.cfg.Kd * de
        return u

    def _pwm_from_u(self, u):
        """Convierte acción PID (grados) a PWM con deadband y slew-rate"""
        mag = abs(u)
        
        # Deadband: no mover si el error es muy pequeño
        if mag < self.cfg.deadband_deg:
            pwm = 0
        else:
            # Mapeo lineal: pwm = PWM_MIN_EFF + pwm_per_degree * error_en_grados
            pwm = int(self.cfg.PWM_MIN_EFF + self.cfg.pwm_per_degree * mag)
            pwm = min(self.cfg.PWM_MAX, pwm)
        
        # Slew-rate: limita cambio de PWM por ciclo
        dp = max(-self.cfg.PWM_SLEW, min(self.cfg.PWM_SLEW, pwm - self._pwm_prev))
        pwm = self._pwm_prev + dp
        self._pwm_prev = pwm
        
        return max(0, pwm)

    def _dir_from_u(self, u):
        """Determina dirección basada en signo de u
        
        u > 0  → Necesita EXTENDER (aumentar roll) → DIR 0
        u < 0  → Necesita RETRAER (disminuir roll) → DIR 1
        """
        return DIR_EXTIENDE if u > 0 else DIR_RETRAE

    def _print_status(self, theta_ref, theta_med, e, pwm, direccion):
        """Imprime estado de control en una línea (formato Control con feedback definitivo)"""
        if pwm == 0:
            dirstr = "STOP"
        else:
            dirstr = "DIR1" if direccion == 1 else "DIR2"
        
        line = (f"Ref={theta_ref:6.1f}°  Medido={theta_med:6.2f}°  "
                f"Error={e:6.2f}°  PWM={pwm:3d}  {dirstr}")
        sys.stdout.write("\r" + line + " " * 10)
        sys.stdout.flush()

    def _send(self, direccion: int, pwm: int):
        """Envía comando dir/PWM por CAN"""
        try:
            import can
            direccion = int(direccion) & 0xFF
            pwm = int(pwm) & 0xFF
            msg = can.Message(
                arbitration_id=self.cfg.ACT_CAN_ID_CMD,
                data=[direccion, pwm],
                is_extended_id=False
            )
            self.bus.send(msg)
        except Exception as e:
            # No imprimir aquí, se maneja en update()
            raise

    def stop(self):
        """Detiene el actuador"""
        try:
            self._send(0, 0)
        except:
            pass

# ======================= SUPERVISOR =======================
class Supervisor:
    """Coordina planificador, percepción, control y actuadores"""
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.plan = CircularTrajectoryPlanner(cfg)
        self.perc = Perception(cfg)
        self.ctrl = Tracker(cfg)
        self.canm = CanManager(cfg)
        self.drive = ODriveCanDriver(cfg, self.canm)
        # MPU6050 SOLO para control de tilt (separado de Perception)
        self.mpu = MPU6050(cfg)
        self.tilt = TiltActuator(cfg, self.canm, self.mpu)
        self.state = "INIT"

    def step(self, now_utc: datetime):
        try:
            if self.state == "INIT":
                self.state = "TRACK"

            if self.state == "TRACK":
                # Planificador: genera referencia circular
                ref = self.plan.step(now_utc)

                # Percepción: obtiene pose + feedback IMU/cámara
                pose, frame_u, marcas, ex = self.perc.get_pose()

                if ref is None:
                    # Noche - detener
                    self.drive.send_wheel_speeds(0.0, 0.0)
                    print("🌙 Noche - sin sol")
                    return

                # **CORRECCIÓN AUTOMÁTICA DEL RAYO** 🎯
                tilt_cmd = ref.tilt_deg  # Valor base del planificador
                
                # Si el rayo está descentrado, aplicar correcciones
                if ex['bright_status'] == "Not centered" and ex['off_px'] > 3.0:
                    # Calcular offsets vertical y horizontal
                    offset_y = ex['bright_pt_y'] - ex['center_y']  # + = rayo abajo, - = rayo arriba
                    offset_x = ex['bright_pt_x'] - ex['center_x']  # + = rayo derecha, - = rayo izq
                    
                    # CORRECCIÓN VERTICAL → TILT
                    # Si el rayo está ARRIBA (offset_y negativo) → AUMENTAR tilt (levantar espejo)
                    # Si el rayo está ABAJO (offset_y positivo) → DISMINUIR tilt (bajar espejo)
                    if abs(offset_y) > 5.0:  # Umbral mínimo 5 píxeles
                        tilt_correction = -offset_y * 0.05  # Factor de corrección (ajustable)
                        tilt_cmd += tilt_correction
                        tilt_cmd = max(self.cfg.tilt_min_deg, min(self.cfg.tilt_max_deg, tilt_cmd))
                
                # ACTUADOR TILT (con corrección aplicada)
                self.tilt.set_tilt_deg(tilt_cmd)
                self.tilt.update()  # Ejecuta un ciclo PID

                # RUEDAS
                if pose is None:
                    # Sin visión - detener
                    self.drive.send_wheel_speeds(0.0, 0.0)
                    print("⚠️ Sin marcas visuales - detenido")
                else:
                    # Control feedback
                    v_cmd, w_cmd = self.ctrl.compute(pose, ref)
                    
                    # **CORRECCIÓN HORIZONTAL DEL RAYO** → AJUSTE DE POSICIÓN
                    # Si el rayo está descentrado horizontalmente, corregir orientación
                    if ex['bright_status'] == "Not centered" and ex['off_px'] > 3.0:
                        offset_x = ex['bright_pt_x'] - ex['center_x']
                        
                        # Si |offset_x| significativo, aplicar corrección angular
                        if abs(offset_x) > 5.0:  # Umbral mínimo 5 píxeles
                            # Rayo a la DERECHA (+) → girar DERECHA (w negativo)
                            # Rayo a la IZQUIERDA (-) → girar IZQUIERDA (w positivo)
                            w_correction = -offset_x * 0.001  # Factor de corrección (ajustable)
                            w_cmd += w_correction
                            w_cmd = max(-self.cfg.w_max, min(self.cfg.w_max, w_cmd))

                    # Conversión diferencial
                    L = self.cfg.track_width
                    vR = v_cmd + 0.5 * L * w_cmd
                    vL = v_cmd - 0.5 * L * w_cmd

                    # Saturación
                    vmax_wheel = self.cfg.v_max + 0.5 * L * self.cfg.w_max
                    scale = max(1.0, max(abs(vR), abs(vL)) / vmax_wheel)
                    vR /= scale
                    vL /= scale

                    self.drive.send_wheel_speeds(vR, vL)

                # Telemetría (incluye estado del rayo y correcciones)
                offset_y = ex['bright_pt_y'] - ex['center_y'] if ex['bright_status'] == "Not centered" else 0
                offset_x = ex['bright_pt_x'] - ex['center_x'] if ex['bright_status'] == "Not centered" else 0
                
                status_icon = "✅" if ex['bright_status'] == "Correct" else "⚠️"
                print(f"{status_icon} r={ref.r:.1f}m  v={v_cmd if pose else 0:.2f}m/s  "
                      f"Rayo: {ex['bright_status']} | off_total={ex['off_px']:.1f}px "
                      f"(↕{offset_y:+.0f}px ↔{offset_x:+.0f}px)  "
                      f"yaw_err={ex['yaw_err']:.1f}°")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"❌ FAULT: {e}")
            self.state = "FAULT"
            self.drive.stop()

    def shutdown(self):
        try:
            self.drive.stop()
        except:
            pass
        try:
            self.tilt.stop()
        except:
            pass
        try:
            self.canm.shutdown()
        except:
            pass

# ======================= MAIN =======================
def main():
    cfg = Config()

    # Ajusta estos parámetros según tu sistema:
    cfg.r_circular = 8.0        # Radio de la circunferencia (metros)
    cfg.tower_height = 3.0      # Altura de la torre (metros)
    cfg.wheel_radius = 0.08     # Radio de ruedas (metros)
    cfg.track_width = 0.50      # Separación entre ruedas (metros)

    print("=" * 70)
    print("  HELIÓSTATO CIRCULAR INTEGRADO")
    print("  Trayectoria circular + Feedback completo (IMUs + Cámara)")
    print("=" * 70)
    print(f"📍 Ubicación: {cfg.lat:.4f}°, {cfg.lon:.4f}°")
    print(f"🔵 Radio circular: {cfg.r_circular} m")
    print(f"🏢 Altura torre: {cfg.tower_height} m")
    print(f"⚙️  CAN: {cfg.can_channel} @ {cfg.can_bitrate} bps")
    print(f"🛞 Ruedas: R={cfg.wheel_radius}m  L={cfg.track_width}m")
    print("Ctrl-C para salir.\n")

    sup = Supervisor(cfg)

    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            sup.step(now_utc)
            time.sleep(cfg.step_s)
    except KeyboardInterrupt:
        print("\n⛔ Apagando...")
    finally:
        sup.shutdown()
        print("✅ Listo.")

if __name__ == "__main__":
    main()
