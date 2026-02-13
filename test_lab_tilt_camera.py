# -*- coding: utf-8 -*-
"""
TEST DE LABORATORIO - CONTROL DE TILT Y CÁMARA (SIN RUEDAS)

Versión simplificada para probar:
- Control de inclinación del espejo (actuador tilt + MPU6050)
- Detección visual del rayo (cámara + patrón)
- Corrección automática de tilt

NO requiere:
- ODrive ni ruedas
- Movimiento del helióstato

Uso:
1. Coloca el helióstato en posición fija apuntando a la torre
2. Configura el tilt objetivo en main()
3. Ejecutar: python test_lab_tilt_camera.py
4. El sistema ajustará automáticamente el tilt para centrar el rayo
"""

from __future__ import annotations
import math, time, struct, sys
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle, Rectangle

# ======================= CONFIGURACIÓN =======================
@dataclass
class Config:
    # Torre
    tower_height: float = 3.0       # Altura torre (m) - solo para referencia
    
    # CAN
    can_channel: str = "can0"
    can_bitrate: int = 500_000
    
    # Actuador tilt
    ACT_CAN_ID_CMD: int = 0x321
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 90.0
    
    # PID Tilt
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
    
    # Detección de rayo
    brightness_threshold: int = 200         # Umbral brillo para "Correct"

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
        print("✅ CAN inicializado")
    
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
        print("✅ Actuador tilt inicializado")
    
    def set_tilt_deg(self, tilt_deg: float):
        self._target_deg = max(self.cfg.tilt_min_deg, 
                              min(self.cfg.tilt_max_deg, float(tilt_deg)))
    
    def get_current_tilt(self) -> Optional[float]:
        """Devuelve el ángulo actual del espejo"""
        pitch, roll = self.mpu.read_pitch_roll()
        if roll is not None:
            return roll * 1.0
        return None
    
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
            dirstr = "STOP" if pwm == 0 else ("EXTEND" if direccion == 0 else "RETRACT")
            line = f"Tilt: Ref={self._target_deg:5.1f}° Act={theta_med:5.1f}° Err={e:5.1f}° PWM={pwm:3d} {dirstr}"
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

# ======================= CÁMARA Y DETECCIÓN =======================
class CameraDetection:
    """Detección simplificada del rayo en la torre"""
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
        
        # BNO055 para localización
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
        
        # Patrón objetivo
        self.REAL_WIDTH = 6.0
        self.REAL_HEIGHT = 10.0
        self.object_points = np.array([[0, 0, 0],
                                       [self.REAL_WIDTH, 0, 0],
                                       [self.REAL_WIDTH, self.REAL_HEIGHT, 0],
                                       [0, self.REAL_HEIGHT, 0]], dtype=np.float32)
        
        print("✅ Cámara inicializada")
    
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
    
    def get_ray_info(self) -> Dict:
        """
        Detecta el patrón y analiza dónde está el rayo
        
        Returns:
            Dict con:
            - detected: bool - si se detectó el patrón
            - centered: bool - si el rayo está centrado
            - offset_y: float - desplazamiento vertical en píxeles
            - offset_x: float - desplazamiento horizontal en píxeles
            - brightness: float - brillo promedio
            - distance_cm: float - distancia a la torre
            - x_m: float - posición X en ENU (m)
            - y_m: float - posición Y en ENU (m)
            - psi: float - orientación en radianes
        """
        frame = self.picam2.capture_array()
        frame_u = self.cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        marcas = self._detectar_marcas(frame_u)
        
        # Leer BNO055 para localización
        yaw_act = self.sensor.euler[0] or 0.0
        pitch_deg = self.sensor.euler[2] or 0.0
        yaw_err = yaw_act - self.yaw_ref_deg
        yaw_err_rad = math.radians(yaw_err)
        
        result = {
            'detected': False,
            'centered': False,
            'offset_y': 0.0,
            'offset_x': 0.0,
            'brightness': 0.0,
            'distance_cm': 0.0,
            'x_m': 0.0,
            'y_m': 0.0,
            'psi': wrap_angle(math.radians(yaw_act)),
            'yaw_deg': yaw_act,
            'frame': frame_u
        }
        
        if len(marcas) == 4:
            pts = self._order4(marcas).astype(np.float32)
            
            # Dibujar marcas
            for (x, y) in pts:
                self.cv2.circle(frame_u, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Calcular distancia
            ok, rvec, tvec = self.cv2.solvePnP(self.object_points, pts, self.camera_matrix,
                                               self.dist_coeffs, flags=self.cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distance_cm = float(tvec[2][0])
                result['distance_cm'] = distance_cm
                
                # Calcular posición XY en ENU
                distance_h_cm = distance_cm * max(0.0, math.cos(math.radians(pitch_deg)))
                X_cm = distance_h_cm * math.sin(yaw_err_rad)
                Y_cm = distance_h_cm * math.cos(yaw_err_rad)
                result['x_m'] = X_cm / 100.0
                result['y_m'] = Y_cm / 100.0
            
            # Centro del patrón
            cx, cy = np.mean(pts, axis=0).astype(int)
            gray = self.cv2.cvtColor(frame_u, self.cv2.COLOR_BGR2GRAY)
            
            # Brillo en el centro
            win = 20
            x1 = max(cx - win//2, 0)
            x2 = min(cx + win//2, gray.shape[1]-1)
            y1 = max(cy - win//2, 0)
            y2 = min(cy + win//2, gray.shape[0]-1)
            
            center_window = gray[y1:y2, x1:x2] if (y2>y1 and x2>x1) else gray[cy:cy+1, cx:cx+1]
            bright_avg = float(np.mean(center_window))
            result['brightness'] = bright_avg
            
            # Verificar si está centrado
            if bright_avg >= self.cfg.brightness_threshold:
                result['centered'] = True
                bright_pt = (cx, cy)
            else:
                # Buscar punto más brillante dentro del patrón
                masked = np.zeros_like(gray)
                self.cv2.fillPoly(masked, [pts.astype(np.int32)], 255)
                masked_gray = self.cv2.bitwise_and(gray, gray, mask=masked)
                _, _, _, maxLoc = self.cv2.minMaxLoc(masked_gray)
                bright_pt = maxLoc
                
                # Calcular offset
                result['offset_y'] = bright_pt[1] - cy
                result['offset_x'] = bright_pt[0] - cx
            
            # Dibujar
            self.cv2.circle(frame_u, (cx, cy), 5, (255, 0, 0), 2)  # Centro azul
            self.cv2.circle(frame_u, bright_pt, 5, (0, 255, 0), -1)  # Punto brillante verde
            
            result['detected'] = True
        
        result['frame'] = frame_u
        
        return result

# ======================= VISUALIZACIÓN MAPA =======================
class MapVisualizer:
    """Visualización en tiempo real de la posición del helióstato"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        plt.ion()  # Modo interactivo
        self.fig = plt.figure(figsize=(18, 7))
        
        # Crear grid de subplots: [mapa | info | cámara]
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.5])
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_camera = self.fig.add_subplot(gs[2])
        
        # Configurar eje del mapa
        self.ax_map.set_xlim(-8, 8)
        self.ax_map.set_ylim(-1, 12)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_xlabel('X (m) - Este →', fontsize=12)
        self.ax_map.set_ylabel('Y (m) - Norte ↑', fontsize=12)
        self.ax_map.set_title('Mapa - Vista desde arriba', fontsize=14, fontweight='bold')
        
        # Torre en el centro
        tower = Circle((0, 0), 0.3, color='red', alpha=0.8, label='Torre')
        self.ax_map.add_patch(tower)
        self.ax_map.plot([0], [0], 'rx', markersize=15, markeredgewidth=3)
        
        # Helióstato (se actualizará)
        self.helio_body = Circle((0, 0), 0.25, color='green', alpha=0.6)
        self.ax_map.add_patch(self.helio_body)
        self.helio_arrow = None
        self.helio_label = self.ax_map.text(0, 0, '', fontsize=10, ha='center')
        
        self.ax_map.legend(loc='upper right')
        
        # Panel de información
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.05, 0.95, '', fontsize=11, 
                                           verticalalignment='top', fontfamily='monospace')
        
        # Panel de cámara
        self.ax_camera.set_title('Detección de Rayo', fontsize=14, fontweight='bold')
        self.ax_camera.axis('off')
        self.camera_image = None
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Dibujar inicial
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
    
    def update(self, ray_info: Dict, current_tilt: float, target_tilt: float):
        """Actualiza la visualización con nueva información"""
        # Actualizar posición del helióstato
        x, y = ray_info['x_m'], ray_info['y_m']
        psi = ray_info['psi']
        
        self.helio_body.set_center((x, y))
        
        # Actualizar flecha de orientación
        if self.helio_arrow:
            self.helio_arrow.remove()
        arrow_len = 0.8
        dx = arrow_len * math.cos(psi)
        dy = arrow_len * math.sin(psi)
        self.helio_arrow = FancyArrow(x, y, dx, dy, width=0.15, 
                                     color='darkgreen', alpha=0.8, zorder=5)
        self.ax_map.add_patch(self.helio_arrow)
        
        # Etiqueta del helióstato
        self.helio_label.set_position((x, y - 0.6))
        self.helio_label.set_text(f'Helióstato\n({x:.2f}, {y:.2f})')
        
        # Panel de información
        status_icon = "✅" if ray_info['centered'] else "⚠️"
        detection_icon = "🎯" if ray_info['detected'] else "❌"
        
        info = f"""
╔════════════════════════════════════════╗
║  ESTADO DEL SISTEMA                    ║
╚════════════════════════════════════════╝

{detection_icon} Detección Patrón: {'SÍ' if ray_info['detected'] else 'NO'}
{status_icon} Rayo Centrado:    {'SÍ' if ray_info['centered'] else 'NO'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POSICIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X (Este):    {x:6.2f} m
Y (Norte):   {y:6.2f} m
Orientación: {math.degrees(psi):6.1f}°
Distancia:   {ray_info['distance_cm']:6.0f} cm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TILT DEL ESPEJO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Actual:      {current_tilt:6.1f}°
Objetivo:    {target_tilt:6.1f}°
Error:       {target_tilt - current_tilt:+6.1f}°

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RAYO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Brillo:      {ray_info['brightness']:6.0f}
Offset Y:    {ray_info['offset_y']:+6.1f} px
Offset X:    {ray_info['offset_x']:+6.1f} px
        """
        
        self.info_text.set_text(info.strip())
        
        # Actualizar imagen de cámara
        if 'frame' in ray_info and ray_info['frame'] is not None:
            frame = ray_info['frame']
            # Convertir BGR a RGB para matplotlib
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]  # BGR -> RGB
            else:
                frame_rgb = frame
            
            if self.camera_image is None:
                self.camera_image = self.ax_camera.imshow(frame_rgb)
            else:
                self.camera_image.set_data(frame_rgb)
        else:
            # Mostrar mensaje si no hay frame
            if self.camera_image is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                self.camera_image = self.ax_camera.imshow(placeholder)
                self.ax_camera.text(0.5, 0.5, 'Esperando imagen...', 
                                   transform=self.ax_camera.transAxes,
                                   ha='center', va='center', fontsize=16, color='white')
        
        # Redibujar
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        plt.close(self.fig)

# ======================= CONTROLADOR PRINCIPAL =======================
class TiltCameraController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.canm = CanManager(cfg)
        self.mpu = MPU6050(cfg)
        self.tilt = TiltActuator(cfg, self.canm, self.mpu)
        self.camera = CameraDetection(cfg)
        self.map_viz = MapVisualizer(cfg)
        print("\n✅ Controlador tilt+cámara inicializado\n")
    
    def run_test(self, target_tilt_deg: float, duration_s: float = 120.0):
        """
        Ejecuta una prueba de tilt sin corrección automática
        
        Args:
            target_tilt_deg: Ángulo objetivo del espejo (°)
            duration_s: Duración de la prueba (s)
        """
        print("=" * 80)
        print(f"🧪 TEST DE TILT Y CÁMARA (SIN MOVIMIENTO)")
        print(f"   Tilt objetivo: {target_tilt_deg:.1f}°")
        print(f"   Duración: {duration_s}s")
        print("   Corrección automática: ❌ OFF (solo monitoreo)")
        print("=" * 80)
        print("\n📌 Coloca el helióstato en posición fija apuntando a la torre")
        print("Presiona Ctrl+C para detener\n")
        
        t_start = time.time()
        
        try:
            while (time.time() - t_start) < duration_s:
                # Leer cámara
                ray_info = self.camera.get_ray_info()
                
                # Aplicar tilt fijo (sin corrección automática)
                self.tilt.set_tilt_deg(target_tilt_deg)
                self.tilt.update()
                
                # Telemetría
                current_tilt = self.tilt.get_current_tilt()
                
                # Actualizar mapa (siempre, aunque no haya detección)
                if current_tilt is not None:
                    self.map_viz.update(ray_info, current_tilt, target_tilt_deg)
                
                if ray_info['detected']:
                    status = "✅ CENTRADO" if ray_info['centered'] else "⚠️ Descentrado"
                    print(f"\n{status} | "
                          f"Pos: ({ray_info['x_m']:.2f}, {ray_info['y_m']:.2f})m | "
                          f"Tilt: {current_tilt:.1f}° (obj: {target_tilt_deg:.1f}°) | "
                          f"Brillo: {ray_info['brightness']:.0f} | "
                          f"Offset Y: {ray_info['offset_y']:+.1f}px")
                else:
                    print(f"\n❌ SIN PATRÓN | Tilt: {current_tilt:.1f}° (obj: {target_tilt_deg:.1f}°)")
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Test interrumpido por usuario")
        finally:
            print("\n🛑 Deteniendo sistema...")
            self.tilt.stop()
            self.canm.shutdown()
            self.map_viz.close()
            print("✅ Test finalizado")

# ======================= MAIN =======================
def main():
    cfg = Config()
    
    # ===== CONFIGURA AQUÍ TU PRUEBA =====
    
    # Tilt objetivo del espejo (fijo, sin corrección automática)
    target_tilt = 30.0  # grados
    
    # Umbral brillo para considerar "centrado"
    cfg.brightness_threshold = 200
    
    # Duración de la prueba
    test_duration = 120.0  # segundos (2 minutos)
    
    # ===== FIN CONFIGURACIÓN =====
    
    print("\n" + "="*80)
    print("  TEST DE LABORATORIO - SOLO TILT Y CÁMARA")
    print("  (No requiere ruedas ni movimiento)")
    print("  (Sin corrección automática - solo monitoreo)")
    print("="*80 + "\n")
    
    controller = TiltCameraController(cfg)
    controller.run_test(target_tilt, duration_s=test_duration)

if __name__ == "__main__":
    main()
