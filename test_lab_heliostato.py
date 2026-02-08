# -*- coding: utf-8 -*-
"""
TEST DE LABORATORIO - HELIÓSTATO CON CORRECCIÓN AUTOMÁTICA DE RAYO

Código simplificado para pruebas en laboratorio.
Permite establecer manualmente:
- Posición objetivo (x, y, ψ) respecto a la torre
- Ángulo tilt objetivo del espejo
- Habilitar/deshabilitar corrección automática del rayo

Uso:
1. Configurar parámetros en main()
2. Ejecutar: python test_lab_heliostato.py
3. Observar correcciones en tiempo real
"""

from __future__ import annotations
import math, time, struct, sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ======================= CONFIGURACIÓN =======================
@dataclass
class Config:
    # Hardware - ajusta según tu setup
    tower_height: float = 3.0       # Altura torre (m)
    track_width: float = 0.50       # Separación ruedas (m)
    wheel_radius: float = 0.08      # Radio rueda (m)
    
    # CAN
    can_channel: str = "can0"
    can_bitrate: int = 500_000
    
    # ODrive
    node_right: int = 1
    node_left: int = 2
    vmax_rev_s_clip: float = 20.0
    sign_R: float = +1.0
    sign_L: float = +1.0
    
    # Actuador tilt
    ACT_CAN_ID_CMD: int = 0x321
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 90.0
    
    # PID Tilt (Control con feedback definitivo)
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    I_clamp: float = 0.6
    deadband_deg: float = 1.0
    PWM_MAX: int = 255
    PWM_MIN_EFF: int = 85
    PWM_SLEW: int = 25
    pwm_per_degree: float = 12.0
    can_send_interval: float = 0.1
    telemetry_hz: float = 5.0
    
    # MPU6050
    mpu_bus: int = 1
    mpu_addr: int = 0x68
    mpu_dt: float = 0.02
    mpu_alpha: float = 0.98
    
    # Control
    v_max: float = 0.3              # Velocidad máxima reducida para lab
    w_max: float = 0.5              # Velocidad angular reducida para lab
    lookahead: float = 0.5          # Lookahead reducido
    k_v: float = 0.4                # Ganancia velocidad
    k_w: float = 1.0                # Ganancia angular
    
    # Corrección de rayo
    enable_ray_correction: bool = True      # Activar/desactivar correcciones
    ray_threshold_px: float = 3.0           # Umbral para activar corrección
    ray_vertical_threshold: float = 5.0     # Umbral vertical
    ray_horizontal_threshold: float = 5.0   # Umbral horizontal
    tilt_correction_factor: float = 0.05    # Factor corrección tilt (°/px)
    w_correction_factor: float = 0.001      # Factor corrección posición (rad/s/px)
    brightness_threshold: int = 200         # Umbral brillo para "Correct"

# ======================= COMANDOS DE PRUEBA =======================
@dataclass
class TestCommand:
    """Comando manual de prueba"""
    x: float            # Posición Este (m)
    y: float            # Posición Norte (m)
    psi: float          # Orientación (rad)
    tilt_deg: float     # Ángulo tilt (°)
    v: float = 0.0      # Velocidad lineal feedforward (m/s)
    w: float = 0.0      # Velocidad angular feedforward (rad/s)

# ======================= UTILIDADES =======================
def wrap_angle(a):
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

# ======================= CAN MANAGER =======================
class CanManager:
    def __init__(self, cfg: Config):
        import can
        self.bus = can.interface.Bus(channel=cfg.can_channel, bustype="socketcan")
    
    def shutdown(self):
        try:
            self.bus.shutdown()
        except:
            pass

# ======================= MPU6050 =======================
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
ACCEL_SCALE = 16384.0
GYRO_SCALE = 131.0

class MPU6050:
    def __init__(self, cfg: Config):
        from smbus2 import SMBus
        self.bus = SMBus(cfg.mpu_bus)
        self.addr = cfg.mpu_addr
        self.dt = cfg.mpu_dt
        self.alpha = cfg.mpu_alpha
        
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)
        time.sleep(0.05)
        self.gyro_bias = self._calibrate_gyro()
        ax, ay, az, *_ = self._read_accel_gyro()
        self.pitch, self.roll = self._accel_to_angles(ax, ay, az)
        print(f"✅ MPU6050 inicializado (bias: x={self.gyro_bias['x']:.2f})")
    
    def _read_word(self, reg):
        hi = self.bus.read_byte_data(self.addr, reg)
        lo = self.bus.read_byte_data(self.addr, reg + 1)
        val = (hi << 8) | lo
        return -((65535 - val) + 1) if val >= 0x8000 else val
    
    def _read_accel_gyro(self):
        ax = self._read_word(ACCEL_XOUT_H) / ACCEL_SCALE
        ay = self._read_word(ACCEL_XOUT_H + 2) / ACCEL_SCALE
        az = self._read_word(ACCEL_XOUT_H + 4) / ACCEL_SCALE
        gx = self._read_word(GYRO_XOUT_H) / GYRO_SCALE
        gy = self._read_word(GYRO_XOUT_H + 2) / GYRO_SCALE
        gz = self._read_word(GYRO_XOUT_H + 4) / GYRO_SCALE
        return ax, ay, az, gx, gy, gz
    
    def _accel_to_angles(self, ax, ay, az):
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
        return pitch, roll
    
    def _calibrate_gyro(self, samples=200):
        print("🔄 Calibrando giroscopio MPU6050...")
        sx = sy = sz = 0.0
        for _ in range(samples):
            _, _, _, gx, gy, gz = self._read_accel_gyro()
            sx += gx; sy += gy; sz += gz
            time.sleep(0.002)
        return {'x': sx/samples, 'y': sy/samples, 'z': sz/samples}
    
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

# ======================= TILT ACTUATOR =======================
DIR_EXTIENDE = 0
DIR_RETRAE = 1

class TiltActuator:
    def __init__(self, cfg: Config, can_mgr: CanManager, mpu: MPU6050):
        self.cfg = cfg
        self.bus = can_mgr.bus
        self.mpu = mpu
        self._target_deg = None
        self._dt = cfg.mpu_dt
        self._ei = 0.0
        self._e_prev = 0.0
        self._pwm_prev = 0
        self._last_can_send = 0.0
        self._last_telemetry = 0.0
    
    def set_tilt_deg(self, tilt_deg: float):
        self._target_deg = max(self.cfg.tilt_min_deg, 
                              min(self.cfg.tilt_max_deg, float(tilt_deg)))
    
    def update(self):
        if self._target_deg is None:
            return
        
        pitch, roll = self.mpu.read_pitch_roll()
        if pitch is None or roll is None:
            return
        
        theta_med = roll * 1.0
        if theta_med < self.cfg.tilt_min_deg or theta_med > self.cfg.tilt_max_deg:
            self._send(0, 0)
            return
        
        e = self._target_deg - theta_med
        u = self._pid(e, self._dt)
        pwm = self._pwm_from_u(u)
        
        if pwm == 0:
            self._send(0, 0)
        else:
            direccion = DIR_EXTIENDE if u > 0 else DIR_RETRAE
            now = time.perf_counter()
            if now - self._last_can_send >= self.cfg.can_send_interval:
                try:
                    self._send(direccion, pwm)
                    self._last_can_send = now
                except:
                    pass
        
        now = time.perf_counter()
        if now - self._last_telemetry >= (1.0 / max(1.0, self.cfg.telemetry_hz)):
            dirstr = "STOP" if pwm == 0 else ("DIR2" if direccion == 0 else "DIR1")
            line = f"Tilt: Ref={self._target_deg:5.1f}° Med={theta_med:5.1f}° e={e:5.1f}° PWM={pwm:3d} {dirstr}"
            sys.stdout.write("\r" + line + " " * 10)
            sys.stdout.flush()
            self._last_telemetry = now
    
    def _pid(self, e, dt):
        self._ei += e * dt
        self._ei = max(-self.cfg.I_clamp, min(self.cfg.I_clamp, self._ei))
        de = (e - self._e_prev) / max(1e-3, dt)
        self._e_prev = e
        return self.cfg.Kp * e + self.cfg.Ki * self._ei + self.cfg.Kd * de
    
    def _pwm_from_u(self, u):
        mag = abs(u)
        if mag < self.cfg.deadband_deg:
            pwm = 0
        else:
            pwm = int(self.cfg.PWM_MIN_EFF + self.cfg.pwm_per_degree * mag)
            pwm = min(self.cfg.PWM_MAX, pwm)
        dp = max(-self.cfg.PWM_SLEW, min(self.cfg.PWM_SLEW, pwm - self._pwm_prev))
        pwm = self._pwm_prev + dp
        self._pwm_prev = pwm
        return max(0, pwm)
    
    def _send(self, direccion: int, pwm: int):
        import can
        msg = can.Message(
            arbitration_id=self.cfg.ACT_CAN_ID_CMD,
            data=[int(direccion) & 0xFF, int(pwm) & 0xFF],
            is_extended_id=False
        )
        self.bus.send(msg)
    
    def stop(self):
        try:
            self._send(0, 0)
        except:
            pass

# ======================= PERCEPCIÓN SIMPLIFICADA =======================
class PerceptionLab:
    """Versión simplificada de percepción para laboratorio"""
    def __init__(self, cfg: Config):
        import cv2, board, busio, adafruit_bno055
        from picamera2 import Picamera2
        
        self.cv2 = cv2
        self.cfg = cfg
        
        # Calibración cámara
        self.camera_matrix = np.array([[1.82798542e+03, 0.0, 5.72342464e+02],
                                       [0.0, 1.82504450e+03, 3.62602479e+02],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([[4.95174274e-03, 2.65701920e+00, -1.43227501e-03,
                                     -1.14715430e-02, -1.37391077e+01]], dtype=np.float32)
        
        # BNO055
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=0x28)
        input("📍 Coloca el robot mirando a la TORRE y pulsa ENTER...")
        self.yaw_ref_deg = self.sensor.euler[0] or 0.0
        print(f"✅ Referencia yaw: {self.yaw_ref_deg:.1f}°")
        
        # Cámara
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        time.sleep(0.2)
        
        # Patrón
        self.REAL_WIDTH = 6.0
        self.REAL_HEIGHT = 10.0
        self.object_points = np.array([[0, 0, 0],
                                       [self.REAL_WIDTH, 0, 0],
                                       [self.REAL_WIDTH, self.REAL_HEIGHT, 0],
                                       [0, self.REAL_HEIGHT, 0]], dtype=np.float32)
        self.FOV_HORIZONTAL = 62.2
        print("✅ Percepción inicializada")
    
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
                    marcas.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
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
        frame = self.picam2.capture_array()
        frame_u = self.cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        marcas = self._detectar_marcas(frame_u)
        
        distance_cm = 0.0
        distance_h_cm = 0.0
        bright_avg = 0.0
        bright_status = "No detection"
        off_px = 0.0
        yaw_act = self.sensor.euler[0] or 0.0
        pitch_deg = self.sensor.euler[2] or 0.0
        yaw_err = yaw_act - self.yaw_ref_deg
        yaw_err_rad = math.radians(yaw_err)
        
        if len(marcas) == 4:
            pts = self._order4(marcas).astype(np.float32)
            for (x, y) in pts:
                self.cv2.circle(frame_u, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            ok, rvec, tvec = self.cv2.solvePnP(self.object_points, pts, self.camera_matrix,
                                               self.dist_coeffs, flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distance_cm = float(tvec[2][0])
            distance_h_cm = distance_cm * max(0.0, math.cos(math.radians(pitch_deg)))
            
            cx, cy = np.mean(pts, axis=0).astype(int)
            gray = self.cv2.cvtColor(frame_u, self.cv2.COLOR_BGR2GRAY)
            
            win = 20
            x1 = max(cx - win//2, 0)
            x2 = min(cx + win//2, gray.shape[1]-1)
            y1 = max(cy - win//2, 0)
            y2 = min(cy + win//2, gray.shape[0]-1)
            
            center_window = gray[y1:y2, x1:x2] if (y2>y1 and x2>x1) else gray[cy:cy+1, cx:cx+1]
            bright_avg = float(np.mean(center_window))
            
            if bright_avg >= self.cfg.brightness_threshold:
                bright_status = "Correct"
                off_px = 0.0
                bright_pt = (cx, cy)
            else:
                bright_status = "Not centered"
                masked = np.zeros_like(gray)
                self.cv2.fillPoly(masked, [pts.astype(np.int32)], 255)
                masked_gray = self.cv2.bitwise_and(gray, gray, mask=masked)
                _, _, _, maxLoc = self.cv2.minMaxLoc(masked_gray)
                bright_pt = maxLoc
                off_px = float(np.linalg.norm(np.array(bright_pt) - np.array([cx, cy])))
            
            self.cv2.circle(frame_u, (cx, cy), 5, (255, 0, 0), 2)
            self.cv2.circle(frame_u, bright_pt, 5, (0, 255, 0), -1)
        else:
            h, w, _ = frame_u.shape
            cx, cy = int(w/2), int(h/2)
            bright_pt = (cx, cy)
        
        X_cm = distance_h_cm * math.sin(yaw_err_rad)
        Y_cm = distance_h_cm * math.cos(yaw_err_rad)
        x_m = X_cm / 100.0
        y_m = Y_cm / 100.0
        psi = wrap_angle(math.radians(yaw_act))
        
        extras = {
            "distance_cm": distance_cm,
            "yaw_act": yaw_act,
            "yaw_err": yaw_err,
            "X_cm": X_cm,
            "Y_cm": Y_cm,
            "bright_avg": bright_avg,
            "bright_status": bright_status,
            "off_px": off_px,
            "bright_pt_x": bright_pt[0],
            "bright_pt_y": bright_pt[1],
            "center_x": cx,
            "center_y": cy
        }
        
        pose = (x_m, y_m, psi) if len(marcas) == 4 else None
        
        # Mostrar frame
        self.cv2.imshow("Lab Test - Detección Rayo", frame_u)
        self.cv2.waitKey(1)
        
        return pose, extras

# ======================= ODRIVE =======================
SET_AXIS_STATE = 0x07
SET_INPUT_VEL = 0x0D
AXIS_CLOSED_LOOP = 8

class ODriveCanDriver:
    def __init__(self, cfg: Config, can_mgr: CanManager):
        self.cfg = cfg
        self.bus = can_mgr.bus
        
        for nid in (cfg.node_right, cfg.node_left):
            msg_id = (nid << 5) | SET_AXIS_STATE
            self.bus.send(self._mk_msg(msg_id, struct.pack("<I", AXIS_CLOSED_LOOP)))
            time.sleep(0.01)
        print("✅ ODrive inicializado")
    
    def _mk_msg(self, arb_id, data=b""):
        import can
        return can.Message(arbitration_id=arb_id, is_extended_id=False, data=data)
    
    def _ms_to_revs(self, v_ms):
        return v_ms / (2.0 * math.pi * self.cfg.wheel_radius * self.cfg.gear_ratio)
    
    def send_wheel_speeds(self, vR_ms, vL_ms):
        vmax = self.cfg.vmax_rev_s_clip
        vR_rev = max(-vmax, min(vmax, self.cfg.sign_R * self._ms_to_revs(vR_ms)))
        vL_rev = max(-vmax, min(vmax, self.cfg.sign_L * self._ms_to_revs(vL_ms)))
        
        msg_id_R = (self.cfg.node_right << 5) | SET_INPUT_VEL
        self.bus.send(self._mk_msg(msg_id_R, struct.pack("<ff", float(vR_rev), 0.0)))
        
        msg_id_L = (self.cfg.node_left << 5) | SET_INPUT_VEL
        self.bus.send(self._mk_msg(msg_id_L, struct.pack("<ff", float(vL_rev), 0.0)))
    
    def stop(self):
        try:
            self.send_wheel_speeds(0.0, 0.0)
        except:
            pass

# ======================= CONTROL =======================
class SimpleTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def compute(self, meas: Tuple[float, float, float], ref: TestCommand):
        xm, ym, psim = meas
        dx = ref.x - xm
        dy = ref.y - ym
        
        ex_r = math.cos(-psim)*dx - math.sin(-psim)*dy
        ey_r = math.sin(-psim)*dx + math.cos(-psim)*dy
        
        Ld = self.cfg.lookahead
        if (ex_r**2 + ey_r**2) > 1e-6:
            scale = Ld / math.hypot(ex_r, ey_r)
            x_look = ex_r * scale
            y_look = ey_r * scale
            kappa = (2.0 * y_look) / (Ld*Ld)
        else:
            kappa = 0.0
        
        e_psi = wrap_angle(ref.psi - psim)
        v_cmd = ref.v + self.cfg.k_v * ex_r
        w_cmd = ref.w + self.cfg.k_w * e_psi + v_cmd * kappa
        
        v_cmd = max(-self.cfg.v_max, min(self.cfg.v_max, v_cmd))
        w_cmd = max(-self.cfg.w_max, min(self.cfg.w_max, w_cmd))
        return v_cmd, w_cmd

# ======================= TEST LAB CONTROLLER =======================
class LabTestController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.canm = CanManager(cfg)
        self.perc = PerceptionLab(cfg)
        self.ctrl = SimpleTracker(cfg)
        self.drive = ODriveCanDriver(cfg, self.canm)
        self.mpu = MPU6050(cfg)
        self.tilt = TiltActuator(cfg, self.canm, self.mpu)
        print("\n✅ Controlador de laboratorio inicializado\n")
    
    def run_test(self, ref: TestCommand, duration_s: float = 30.0):
        """Ejecuta una prueba con el comando especificado"""
        print("=" * 70)
        print(f"🧪 PRUEBA DE LABORATORIO")
        print(f"   Posición objetivo: x={ref.x:.2f}m, y={ref.y:.2f}m, ψ={math.degrees(ref.psi):.1f}°")
        print(f"   Tilt objetivo: {ref.tilt_deg:.1f}°")
        print(f"   Duración: {duration_s}s")
        print(f"   Corrección automática: {'✅ ON' if self.cfg.enable_ray_correction else '❌ OFF'}")
        print("=" * 70)
        print("\nPresiona Ctrl+C para detener\n")
        
        t_start = time.time()
        
        try:
            while (time.time() - t_start) < duration_s:
                # Percepción
                pose, ex = self.perc.get_pose()
                
                # Corrección de tilt
                tilt_cmd = ref.tilt_deg
                
                if self.cfg.enable_ray_correction and ex['bright_status'] == "Not centered" and ex['off_px'] > self.cfg.ray_threshold_px:
                    offset_y = ex['bright_pt_y'] - ex['center_y']
                    
                    if abs(offset_y) > self.cfg.ray_vertical_threshold:
                        tilt_correction = -offset_y * self.cfg.tilt_correction_factor
                        tilt_cmd += tilt_correction
                        tilt_cmd = max(self.cfg.tilt_min_deg, min(self.cfg.tilt_max_deg, tilt_cmd))
                
                self.tilt.set_tilt_deg(tilt_cmd)
                self.tilt.update()
                
                # Control de ruedas
                if pose is None:
                    self.drive.send_wheel_speeds(0.0, 0.0)
                    print("\n⚠️  Sin marcas - detenido")
                else:
                    v_cmd, w_cmd = self.ctrl.compute(pose, ref)
                    
                    # Corrección horizontal
                    if self.cfg.enable_ray_correction and ex['bright_status'] == "Not centered" and ex['off_px'] > self.cfg.ray_threshold_px:
                        offset_x = ex['bright_pt_x'] - ex['center_x']
                        
                        if abs(offset_x) > self.cfg.ray_horizontal_threshold:
                            w_correction = -offset_x * self.cfg.w_correction_factor
                            w_cmd += w_correction
                            w_cmd = max(-self.cfg.w_max, min(self.cfg.w_max, w_cmd))
                    
                    # Diferencial
                    L = self.cfg.track_width
                    vR = v_cmd + 0.5 * L * w_cmd
                    vL = v_cmd - 0.5 * L * w_cmd
                    
                    vmax_wheel = self.cfg.v_max + 0.5 * L * self.cfg.w_max
                    scale = max(1.0, max(abs(vR), abs(vL)) / vmax_wheel)
                    vR /= scale
                    vL /= scale
                    
                    self.drive.send_wheel_speeds(vR, vL)
                    
                    # Telemetría
                    offset_y = ex['bright_pt_y'] - ex['center_y'] if ex['bright_status'] == "Not centered" else 0
                    offset_x = ex['bright_pt_x'] - ex['center_x'] if ex['bright_status'] == "Not centered" else 0
                    status_icon = "✅" if ex['bright_status'] == "Correct" else "⚠️"
                    
                    print(f"\n{status_icon} Pos: x={pose[0]:.2f}m y={pose[1]:.2f}m | "
                          f"Rayo: {ex['bright_status']} | off={ex['off_px']:.1f}px (↕{offset_y:+.0f} ↔{offset_x:+.0f}) | "
                          f"v={v_cmd:.2f}m/s")
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Test interrumpido por usuario")
        finally:
            print("\n🛑 Deteniendo sistema...")
            self.drive.stop()
            self.tilt.stop()
            self.canm.shutdown()
            print("✅ Test finalizado")

# ======================= MAIN =======================
def main():
    cfg = Config()
    
    # ===== CONFIGURA AQUÍ TU PRUEBA =====
    
    # Parámetros de hardware
    cfg.tower_height = 3.0      # Altura de tu torre (m)
    cfg.wheel_radius = 0.08     # Radio de tus ruedas (m)
    cfg.track_width = 0.50      # Separación entre ruedas (m)
    
    # Comando de prueba - AJUSTA ESTOS VALORES
    test_cmd = TestCommand(
        x=2.0,              # Posición Este respecto a torre (m)
        y=5.0,              # Posición Norte respecto a torre (m)
        psi=math.radians(0),  # Orientación (rad) - 0 = mirando norte
        tilt_deg=30.0,      # Ángulo tilt del espejo (°)
        v=0.0,              # Velocidad feedforward (m/s)
        w=0.0               # Velocidad angular feedforward (rad/s)
    )
    
    # Configuración de corrección automática
    cfg.enable_ray_correction = True        # True = corrección ON, False = OFF
    cfg.tilt_correction_factor = 0.05       # Factor corrección tilt (ajustar)
    cfg.w_correction_factor = 0.001         # Factor corrección posición (ajustar)
    cfg.ray_threshold_px = 3.0              # Umbral activación corrección
    cfg.brightness_threshold = 200          # Umbral brillo para "Correct"
    
    # Duración de la prueba
    test_duration = 60.0  # segundos
    
    # ===== FIN CONFIGURACIÓN =====
    
    controller = LabTestController(cfg)
    controller.run_test(test_cmd, duration_s=test_duration)

if __name__ == "__main__":
    main()
