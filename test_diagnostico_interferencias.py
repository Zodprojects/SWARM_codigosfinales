# -*- coding: utf-8 -*-
"""
DIAGNÓSTICO DE INTERFERENCIAS MPU6050 vs MOTORES ODRIVE
========================================================
Basado en la arquitectura de Camara + feddback verde +odrive.py

Se prueba:
  TEST 1 — BUS I2C: ¿detecta ambos sensores antes de encender motores?
  TEST 2 — BASELINE MPU6050: estabilidad de lectura sin motores
  TEST 3 — BASELINE BNO055:  estabilidad de lectura sin motores
  TEST 4 — ACTIVAR ODRIVE (closed-loop, sin mover)
  TEST 5 — MOTORES EN MARCHA: lectura simultánea MPU6050 + BNO055
  TEST 6 — ANÁLISIS: tipo de fallo, si se recupera solo, correlación con velocidad

Uso:
  1. Conecta todo el hardware (ODrive + motores + MPU6050 + BNO055)
  2. Deja los motores en reposo (sin girar)
  3. Ejecuta: python test_diagnostico_interferencias.py
  4. Sigue las instrucciones en pantalla
"""

from __future__ import annotations
import math
import time
import struct
import sys
import statistics
from dataclasses import dataclass, field
from typing import Optional
import threading

# ─────────────────────────── CONFIGURACIÓN ───────────────────────────
@dataclass
class Config:
    # I2C
    i2c_bus:      int   = 1
    mpu_addr:     int   = 0x68
    bno_addr:     int   = 0x28   # 0x28 o 0x29 según pin ADR
    # CAN / ODrive  (igual que en el código principal)
    can_channel:  str   = "can0"
    can_bitrate:  int   = 500_000
    node_right:   int   = 1
    node_left:    int   = 2
    # Velocidad de prueba de los motores (rev/s)
    test_speed_low:  float = 2.0
    test_speed_high: float = 8.0
    # Duración de cada fase de motores (segundos)
    phase_duration: float = 8.0
    # Frecuencia de muestreo del bucle de diagnóstico (Hz)
    sample_hz: float = 20.0

CFG = Config()

# ─────────────────────────── REGISTROS MPU6050 ───────────────────────────
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACCEL_SCALE  = 16384.0
GYRO_SCALE   = 131.0

# ─────────────────────────── ODRIVE CANSimple IDs ────────────────────────
SET_AXIS_STATE        = 0x07
SET_INPUT_VEL         = 0x0D
GET_ENCODER_ESTIMATES = 0x09
AXIS_IDLE             = 1
AXIS_CLOSEDLOOP       = 8

# ─────────────────────────── COLORES CONSOLA ─────────────────────────────
RED   = "\033[91m"
GRN   = "\033[92m"
YEL   = "\033[93m"
BLU   = "\033[94m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(msg):  print(f"  {GRN}✔{RESET} {msg}")
def err(msg): print(f"  {RED}✘{RESET} {RED}{msg}{RESET}")
def warn(msg):print(f"  {YEL}⚠{RESET} {msg}")
def hdr(msg): print(f"\n{BOLD}{BLU}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")
def sep():    print("─" * 60)

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS I2C
# ═══════════════════════════════════════════════════════════════════════════

def i2c_device_present(bus, addr: int) -> bool:
    try:
        bus.read_byte(addr)
        return True
    except Exception:
        return False

def mpu_read_word(bus, addr, reg):
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    return -((65535 - val) + 1) if val >= 0x8000 else val

def mpu_read_accel(bus, addr):
    ax = mpu_read_word(bus, addr, ACCEL_XOUT_H)     / ACCEL_SCALE
    ay = mpu_read_word(bus, addr, ACCEL_XOUT_H + 2) / ACCEL_SCALE
    az = mpu_read_word(bus, addr, ACCEL_XOUT_H + 4) / ACCEL_SCALE
    return ax, ay, az

def mpu_init(bus, addr):
    bus.write_byte_data(addr, PWR_MGMT_1, 0x00)
    time.sleep(0.05)

def bno_read_euler(sensor):
    """Lee euler angles de la BNO055 (adafruit driver), devuelve (yaw, roll, pitch) o None."""
    try:
        e = sensor.euler
        if e is None or e[0] is None:
            return None
        return e   # (yaw, roll, pitch)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  ODRIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def odrive_set_state(can_bus, node_id: int, state: int):
    import can as canlib
    arb = (node_id << 5) | SET_AXIS_STATE
    msg = canlib.Message(arbitration_id=arb, is_extended_id=False,
                         data=struct.pack("<I", state))
    can_bus.send(msg)

def odrive_set_vel(can_bus, node_id: int, vel_rev_s: float):
    import can as canlib
    arb = (node_id << 5) | SET_INPUT_VEL
    msg = canlib.Message(arbitration_id=arb, is_extended_id=False,
                         data=struct.pack("<ff", vel_rev_s, 0.0))
    can_bus.send(msg)

def odrive_stop_all(can_bus):
    for nid in (CFG.node_right, CFG.node_left):
        try:
            odrive_set_vel(can_bus, nid, 0.0)
            time.sleep(0.02)
            odrive_set_state(can_bus, nid, AXIS_IDLE)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════
#  CLASE DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    name:          str
    mpu_reads_ok:  int = 0
    mpu_reads_err: int = 0
    mpu_oserrors:  list = field(default_factory=list)
    mpu_anomalies: int = 0
    bno_reads_ok:  int = 0
    bno_reads_err: int = 0
    mpu_recovered: int = 0      # veces que respondió tras un fallo
    mpu_pitch_vals: list = field(default_factory=list)
    bno_yaw_vals:   list = field(default_factory=list)
    motor_speed:    float = 0.0
    duration_s:     float = 0.0

    @property
    def mpu_error_rate(self) -> float:
        total = self.mpu_reads_ok + self.mpu_reads_err
        return (self.mpu_reads_err / total * 100) if total else 0.0

    @property
    def bno_error_rate(self) -> float:
        total = self.bno_reads_ok + self.bno_reads_err
        return (self.bno_reads_err / total * 100) if total else 0.0

    @property
    def mpu_pitch_std(self) -> float:
        return statistics.stdev(self.mpu_pitch_vals) if len(self.mpu_pitch_vals) > 1 else 0.0

# ═══════════════════════════════════════════════════════════════════════════
#  BUCLE DE MUESTREO
# ═══════════════════════════════════════════════════════════════════════════

def sampling_loop(i2c_bus, bno_sensor, result: PhaseResult,
                  duration: float, stop_event: threading.Event):
    """
    Corre en thread separado. Muestrea MPU6050 y BNO055 a CFG.sample_hz.
    Detecta:
      - OSError (pérdida de dispositivo en bus)
      - Valores anómalos (aceleración fuera de rango físico)
      - Recuperación espontánea
    """
    dt = 1.0 / CFG.sample_hz
    t0 = time.time()
    prev_mpu_ok = True

    while not stop_event.is_set() and (time.time() - t0) < duration:
        t_loop = time.time()

        # ── MPU6050 ──
        mpu_ok = False
        try:
            ax, ay, az = mpu_read_accel(i2c_bus, CFG.mpu_addr)
            # Detección de valor anómalo (imposible físicamente)
            mag = math.sqrt(ax*ax + ay*ay + az*az)
            if mag < 0.1 or mag > 20.0:
                result.mpu_anomalies += 1
                err(f"[{time.time()-t0:.2f}s] MPU6050 valor anómalo |a|={mag:.2f}g")
            else:
                pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
                result.mpu_pitch_vals.append(pitch)
                result.mpu_reads_ok += 1
                mpu_ok = True
                # Recuperación tras fallo
                if not prev_mpu_ok:
                    result.mpu_recovered += 1
                    warn(f"[{time.time()-t0:.2f}s] MPU6050 SE RECUPERÓ SOLO")
        except OSError as e:
            result.mpu_reads_err += 1
            errno_str = str(e)
            result.mpu_oserrors.append(errno_str)
            err(f"[{time.time()-t0:.2f}s] MPU6050 OSError: {errno_str}")
        except Exception as e:
            result.mpu_reads_err += 1
            err(f"[{time.time()-t0:.2f}s] MPU6050 {type(e).__name__}: {e}")

        prev_mpu_ok = mpu_ok

        # ── BNO055 ──
        euler = bno_read_euler(bno_sensor)
        if euler is not None:
            result.bno_reads_ok += 1
            result.bno_yaw_vals.append(euler[0])
        else:
            result.bno_reads_err += 1
            err(f"[{time.time()-t0:.2f}s] BNO055 lectura fallida")

        # Mantener frecuencia
        elapsed = time.time() - t_loop
        time.sleep(max(0.0, dt - elapsed))

    result.duration_s = time.time() - t0

# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN DE IMPRESIÓN DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def print_phase_result(r: PhaseResult):
    sep()
    print(f"  Fase: {BOLD}{r.name}{RESET}  |  Velocidad motores: {r.motor_speed:.1f} rev/s  |  Duración: {r.duration_s:.1f}s")
    sep()

    # MPU6050
    total_mpu = r.mpu_reads_ok + r.mpu_reads_err
    mpu_color = RED if r.mpu_error_rate > 5 else (YEL if r.mpu_error_rate > 0 else GRN)
    print(f"  MPU6050  | Lecturas OK: {r.mpu_reads_ok:4d} | Errores: {r.mpu_reads_err:4d} | "
          f"Tasa error: {mpu_color}{r.mpu_error_rate:.1f}%{RESET} | "
          f"Anómalos: {r.mpu_anomalies} | Recuperaciones: {r.mpu_recovered}")
    if r.mpu_oserrors:
        tipos = {}
        for e in r.mpu_oserrors:
            tipos[e] = tipos.get(e, 0) + 1
        for tipo, cnt in tipos.items():
            print(f"           -> {cnt}x  {tipo}")
    if r.mpu_pitch_vals:
        print(f"           -> Pitch: media={statistics.mean(r.mpu_pitch_vals):.2f}°  "
              f"std={r.mpu_pitch_std:.3f}°  "
              f"min={min(r.mpu_pitch_vals):.2f}°  max={max(r.mpu_pitch_vals):.2f}°")

    # BNO055
    bno_color = RED if r.bno_error_rate > 5 else (YEL if r.bno_error_rate > 0 else GRN)
    print(f"  BNO055   | Lecturas OK: {r.bno_reads_ok:4d} | Errores: {r.bno_reads_err:4d} | "
          f"Tasa error: {bno_color}{r.bno_error_rate:.1f}%{RESET}")
    sep()

# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO FINAL
# ═══════════════════════════════════════════════════════════════════════════

def print_diagnosis(results: list[PhaseResult]):
    hdr("DIAGNÓSTICO FINAL")

    baseline = next((r for r in results if "REPOSO" in r.name.upper()), None)

    for r in results:
        if "REPOSO" in r.name.upper():
            continue
        print(f"\n  Fase [{r.name}]:")

        # ¿Empeoró respecto al baseline?
        if baseline:
            delta = r.mpu_error_rate - baseline.mpu_error_rate
            if delta > 2.0:
                err(f"  MPU6050 empeoró {delta:.1f}% vs reposo => INTERFERENCIA CONFIRMADA")
            else:
                ok("  MPU6050 sin cambio significativo vs reposo")

        if r.mpu_error_rate > 30:
            err("  MPU6050 FALLO GRAVE (>30% errores). Solución hardware necesaria.")
        elif r.mpu_error_rate > 5:
            warn("  MPU6050 FALLO MODERADO (>5%). Ferritas/condensadores pueden ayudar.")
        else:
            ok("  MPU6050 estable con motores.")

        if r.mpu_recovered > 0:
            ok(f"  MPU6050 se recuperó {r.mpu_recovered} veces => bus NO queda bloqueado permanentemente")
            ok("  => Posible solución: reintentos automáticos en software")
        elif r.mpu_reads_err > 0 and r.mpu_recovered == 0:
            err("  MPU6050 NO se recuperó => bus I2C queda bloqueado => necesita reset hardware")

        if r.bno_error_rate > 2.0:
            warn("  BNO055 también tuvo errores => problema es el BUS I2C (no el sensor)")
        else:
            ok("  BNO055 sin errores => la MPU6050 es más susceptible al ruido")

    # Análisis de progresión por velocidad
    motor_phases = [r for r in results if r.motor_speed > 0]
    if len(motor_phases) >= 2:
        print("\n  Correlación velocidad → tasa de error MPU6050:")
        for r in sorted(motor_phases, key=lambda x: x.motor_speed):
            bar = "█" * int(r.mpu_error_rate / 2)
            print(f"    {r.motor_speed:5.1f} rev/s  |{bar:<30}| {r.mpu_error_rate:.1f}%")
        rates = [r.mpu_error_rate for r in sorted(motor_phases, key=lambda x: x.motor_speed)]
        if rates[-1] > rates[0] + 5:
            warn("  La tasa de error AUMENTA con la velocidad => EMI de bobinados/PWM")
        else:
            warn("  La tasa de error NO correlaciona con velocidad => fuente de ruido constante")

    # Conclusión
    print(f"\n{BOLD}  CONCLUSIÓN:{RESET}")
    all_motor = [r for r in results if r.motor_speed > 0]
    if not all_motor:
        warn("  No se ejecutaron fases con motores (ODrive no disponible)")
        return

    max_err = max(r.mpu_error_rate for r in all_motor)
    if max_err == 0:
        ok("  No hay interferencias detectadas. El sistema es estable.")
    elif max_err < 5:
        warn("  Interferencias leves. Probar ferritas + condensadores 100pF.")
    elif max_err < 30:
        warn("  Interferencias moderadas. Ferritas pueden no ser suficientes.")
        warn("  Considera: cables apantallados, filtro en VCC de MPU6050, o BNO055 en el espejo.")
    else:
        err("  Interferencias severas. Necesitas solución hardware (BNO055 en espejo o SPI).")
        if all(r.mpu_recovered == 0 for r in all_motor if r.mpu_reads_err > 0):
            err("  Bus I2C queda permanentemente bloqueado. Reset de software NO ayuda.")

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═'*60}")
    print("  DIAGNÓSTICO INTERFERENCIAS MPU6050 / ODrive")
    print(f"{'═'*60}{RESET}\n")
    print("  Este script evalúa si los motores ODrive interfieren con el")
    print("  bus I2C de la MPU6050, usando la misma arquitectura hardware")
    print("  que el código principal de control del helióstato.\n")

    results: list[PhaseResult] = []
    can_bus = None

    # ── Importaciones de hardware ──
    try:
        from smbus2 import SMBus
        i2c_bus = SMBus(CFG.i2c_bus)
    except ImportError:
        err("smbus2 no instalado. Ejecuta: pip install smbus2")
        sys.exit(1)
    except Exception as e:
        err(f"No se pudo abrir bus I2C: {e}")
        sys.exit(1)

    try:
        import board, busio, adafruit_bno055
        i2c_hw = busio.I2C(board.SCL, board.SDA)
        bno = adafruit_bno055.BNO055_I2C(i2c_hw, address=CFG.bno_addr)
    except ImportError:
        err("Librerías BNO055 no instaladas. Ejecuta: pip install adafruit-circuitpython-bno055")
        sys.exit(1)
    except Exception as e:
        err(f"No se pudo inicializar BNO055: {e}")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────
    # TEST 1 — ESCANER DE BUS I2C
    # ──────────────────────────────────────────────────────────────
    hdr("TEST 1 — Escáner de bus I2C")
    print("  Buscando dispositivos I2C antes de encender motores...\n")

    found = []
    for addr in range(0x08, 0x78):
        if i2c_device_present(i2c_bus, addr):
            found.append(addr)

    if found:
        ok(f"Dispositivos encontrados: {[hex(a) for a in found]}")
    else:
        err("No se encontró ningún dispositivo I2C. Verifica conexiones.")
        sys.exit(1)

    mpu_present = CFG.mpu_addr in found
    bno_present = CFG.bno_addr in found

    if mpu_present:
        ok(f"MPU6050 detectada en 0x{CFG.mpu_addr:02X}")
    else:
        err(f"MPU6050 NO encontrada en 0x{CFG.mpu_addr:02X}")

    if bno_present:
        ok(f"BNO055 detectada en 0x{CFG.bno_addr:02X}")
    else:
        warn(f"BNO055 no encontrada en 0x{CFG.bno_addr:02X} (prueba 0x29)")
        CFG.bno_addr = 0x29
        if i2c_device_present(i2c_bus, 0x29):
            ok("BNO055 encontrada en 0x29, ajustando config.")
            bno_present = True
        else:
            err("BNO055 no encontrada en ninguna dirección. Verifica conexiones.")

    if not mpu_present:
        print("\n  ¿Continuar sin MPU6050? (s/n): ", end="")
        if input().strip().lower() != "s":
            sys.exit(0)

    # Inicializar MPU6050
    if mpu_present:
        try:
            mpu_init(i2c_bus, CFG.mpu_addr)
            ok("MPU6050 inicializada (salida de sleep mode)")
        except Exception as e:
            err(f"Error inicializando MPU6050: {e}")

    # ──────────────────────────────────────────────────────────────
    # TEST 2 — BASELINE (SIN MOTORES)
    # ──────────────────────────────────────────────────────────────
    hdr("TEST 2 — Baseline sin motores")
    print("  Medición de referencia con todo en reposo (sin ODrive, sin motores).\n")
    print(f"  {YEL}¿Continuar con TEST 2? (s/n):{RESET} ", end="")
    if input().strip().lower() != "s":
        warn("TEST 2 saltado. No se puede continuar sin baseline.")
        sys.exit(0)
    print()

    r_baseline = PhaseResult(name="REPOSO (sin motores)", motor_speed=0.0)
    stop_evt = threading.Event()
    t = threading.Thread(target=sampling_loop,
                         args=(i2c_bus, bno, r_baseline, CFG.phase_duration, stop_evt))
    t.start()

    print(f"  Muestreando {CFG.phase_duration:.0f}s a {CFG.sample_hz:.0f}Hz...")
    t.join()
    results.append(r_baseline)
    print_phase_result(r_baseline)

    # ──────────────────────────────────────────────────────────────
    # TEST 3 — ODRIVE ENCENDIDO (CLOSED-LOOP, MOTORES PARADOS)
    # ──────────────────────────────────────────────────────────────
    hdr("TEST 3 — ODrive en closed-loop (sin mover)")
    print("  Este test pone el ODrive en CLOSED-LOOP con velocidad 0.")
    print("  Los motores NO girarán, pero el ODrive se activará eléctricamente.\n")
    print(f"  {YEL}¿Continuar con TEST 3? (s/n):{RESET} ", end="")
    if input().strip().lower() != "s":
        warn("TEST 3 saltado por el usuario. Fin del diagnóstico.")
        print_diagnosis(results)
        return
    print()

    odrive_available = False
    try:
        import can as canlib
        can_bus = canlib.interface.Bus(channel=CFG.can_channel, bustype="socketcan")
        for nid in (CFG.node_right, CFG.node_left):
            odrive_set_state(can_bus, nid, AXIS_CLOSEDLOOP)
            time.sleep(0.05)
            odrive_set_vel(can_bus, nid, 0.0)
        ok("ODrive en closed-loop, velocidad 0")
        odrive_available = True
    except ImportError:
        warn("python-can no instalado. Saltando tests con ODrive.")
    except Exception as e:
        warn(f"No se pudo inicializar ODrive: {e}. Saltando tests con motores.")

    if odrive_available:
        r_idle = PhaseResult(name="ODRIVE ENCENDIDO (vel=0)", motor_speed=0.0)
        stop_evt = threading.Event()
        t = threading.Thread(target=sampling_loop,
                             args=(i2c_bus, bno, r_idle, CFG.phase_duration, stop_evt))
        t.start()
        print(f"  Muestreando {CFG.phase_duration:.0f}s...")
        t.join()
        results.append(r_idle)
        print_phase_result(r_idle)

        # ──────────────────────────────────────────────────────────
        # TEST 4 — MOTORES A VELOCIDAD BAJA
        # ──────────────────────────────────────────────────────────
        hdr(f"TEST 4 — Motores girando a velocidad BAJA ({CFG.test_speed_low} rev/s)")
        print(f"  Los motores girarán a {CFG.test_speed_low} rev/s. Asegúrate de que:")
        print("    - El robot está sobre el suelo o elevado de forma segura")
        print("    - No hay obstáculos en las ruedas")
        print("    - Tienes acceso al corte de emergencia\n")
        print(f"  {YEL}¿Continuar con TEST 4? (s/n):{RESET} ", end="")
        if input().strip().lower() != "s":
            warn("TEST 4 saltado. Pasando a diagnóstico con datos actuales.")
            odrive_stop_all(can_bus)
            print_diagnosis(results)
            return
        print()

        for nid in (CFG.node_right, CFG.node_left):
            odrive_set_vel(can_bus, nid, CFG.test_speed_low)
        time.sleep(0.5)  # Dejar que arranquen

        r_low = PhaseResult(name=f"MOTORES VEL. BAJA ({CFG.test_speed_low} rev/s)",
                            motor_speed=CFG.test_speed_low)
        stop_evt = threading.Event()
        t = threading.Thread(target=sampling_loop,
                             args=(i2c_bus, bno, r_low, CFG.phase_duration, stop_evt))
        t.start()
        print(f"  Muestreando {CFG.phase_duration:.0f}s con motores girando...")
        t.join()

        # Parar motores
        for nid in (CFG.node_right, CFG.node_left):
            odrive_set_vel(can_bus, nid, 0.0)
        results.append(r_low)
        print_phase_result(r_low)

        # ──────────────────────────────────────────────────────────
        # TEST 5 — MOTORES A VELOCIDAD ALTA
        # ──────────────────────────────────────────────────────────
        hdr(f"TEST 5 — Motores girando a velocidad ALTA ({CFG.test_speed_high} rev/s)")
        print(f"  Los motores girarán a {CFG.test_speed_high} rev/s (máxima carga EMI).")
        print("    - Confirma que el entorno sigue siendo seguro")
        print("    - Este test genera más ruido eléctrico que el anterior\n")
        print(f"  {YEL}¿Continuar con TEST 5? (s/n):{RESET} ", end="")
        if input().strip().lower() != "s":
            warn("TEST 5 saltado. Diagnóstico con datos hasta ahora.")
            odrive_stop_all(can_bus)
            print_diagnosis(results)
            return
        print()

        for nid in (CFG.node_right, CFG.node_left):
            odrive_set_vel(can_bus, nid, CFG.test_speed_high)
        time.sleep(0.5)

        r_high = PhaseResult(name=f"MOTORES VEL. ALTA ({CFG.test_speed_high} rev/s)",
                             motor_speed=CFG.test_speed_high)
        stop_evt = threading.Event()
        t = threading.Thread(target=sampling_loop,
                             args=(i2c_bus, bno, r_high, CFG.phase_duration, stop_evt))
        t.start()
        print(f"  Muestreando {CFG.phase_duration:.0f}s con motores a alta velocidad...")
        t.join()

        # Parar motores y poner en idle
        odrive_stop_all(can_bus)
        ok("ODrive detenido y puesto en idle")
        results.append(r_high)
        print_phase_result(r_high)

    # ──────────────────────────────────────────────────────────────
    # TEST 6 — TEST DE RECUPERACIÓN (solo si hubo errores)
    # ──────────────────────────────────────────────────────────────
    motor_results = [r for r in results if r.motor_speed > 0]
    if motor_results and any(r.mpu_reads_err > 0 for r in motor_results):
        hdr("TEST 6 — Test de recuperación del bus I2C")
        print("  Verificando si el bus I2C se puede recuperar por software...\n")

        bus_ok = i2c_device_present(i2c_bus, CFG.mpu_addr)
        if bus_ok:
            ok("MPU6050 responde tras detener motores => el bus se recupera solo al parar")
        else:
            err("MPU6050 NO responde con motores parados => bus I2C está BLOQUEADO")
            print("\n  Intentando reinicializar MPU6050...")
            try:
                mpu_init(i2c_bus, CFG.mpu_addr)
                time.sleep(0.1)
                if i2c_device_present(i2c_bus, CFG.mpu_addr):
                    ok("Re-init de MPU6050 funcionó => reiniciar el objeto resuelve el bloqueo")
                else:
                    err("Re-init falló => necesita reset físico (power-cycle del sensor)")
            except Exception as e:
                err(f"Error en re-init: {e}")

    # ──────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────
    print_diagnosis(results)

    # Guardar log
    log_path = "diagnostico_interferencias_log.txt"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("DIAGNÓSTICO INTERFERENCIAS MPU6050 vs ODrive\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for r in results:
                f.write(f"Fase: {r.name}\n")
                f.write(f"  motor_speed={r.motor_speed} rev/s\n")
                f.write(f"  mpu_ok={r.mpu_reads_ok}  mpu_err={r.mpu_reads_err}  "
                        f"error_rate={r.mpu_error_rate:.2f}%\n")
                f.write(f"  mpu_anomalies={r.mpu_anomalies}  mpu_recovered={r.mpu_recovered}\n")
                f.write(f"  bno_ok={r.bno_reads_ok}  bno_err={r.bno_reads_err}  "
                        f"bno_error_rate={r.bno_error_rate:.2f}%\n")
                if r.mpu_oserrors:
                    from collections import Counter
                    for tipo, cnt in Counter(r.mpu_oserrors).items():
                        f.write(f"  oserror: {cnt}x {tipo}\n")
                f.write("\n")
        ok(f"Log guardado en: {log_path}")
    except Exception as e:
        warn(f"No se pudo guardar el log: {e}")

    # Cleanup
    if can_bus:
        try:
            can_bus.shutdown()
        except Exception:
            pass
    try:
        i2c_bus.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YEL}  Interrumpido por el usuario.{RESET}")
        sys.exit(0)
    except Exception as e:
        err(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
