import glob
import math
import re
import time

import serial
from Joysticks import F710Controller

_CONNECTION_RETRY_ATTEMPTS = 3
_BAUD_RATE                 = 115200
_LOOP_PERIOD_S             = 0.05     # 20 Hz, well inside the Arduino's 500 ms failsafe

_AXIS_RIGHT_X = 3                      # ABS_RX
_AXIS_RIGHT_Y = 4                      # ABS_RY
_DEADZONE     = 0.08                   # stick drift at rest stays below this
_MAX_COMMAND  = 204                    # matches MAX_PWM on the Arduino

_TICKS_PER_REV  = 660                  # 11 PPR x 30:1 gearbox x2 decoding
_WHEEL_RADIUS_M = 0.045

_TELEMETRY = re.compile(rb"FR:(-?\d+) FL:(-?\d+) us:(\d+)")


# ============================================================ connection
def _connectArduino(baudRate, retryAttempts):
    print("! Searching for Arduino on serial port /dev/ttyACM*...")
    for _ in range(retryAttempts):
        for port in glob.glob("/dev/ttyACM*"):
            try:
                arduino = serial.Serial(port, baudRate, timeout=0, write_timeout=1)
            except Exception as e:
                print(f"    - Found {port} but couldn't open it: {e}")
                continue
            print(f"    - Connected to Arduino on {port} at {baudRate} baud")
            time.sleep(2)                  # opening the port resets the Mega; wait out the bootloader
            arduino.reset_input_buffer()   # discard anything sent while booting
            return arduino
        print("    - Cannot find Arduino. Retrying")
        time.sleep(2)
    raise RuntimeError(f"! ERROR: could not connect to Arduino after {retryAttempts} attempts")


# ============================================================ read sensors
def readJoystick(joystick):
    leftX, _  = joystick.getNormalisedLeftJoystickVector()    # left stick X  -> rotation
    _, rightY = joystick.getNormalisedRightJoystickVector()   # right stick Y -> linear
    return leftX, -rightY


def readEncoders(arduino, rxBuffer):
    rxBuffer.extend(arduino.read(arduino.in_waiting))
    *lines, tail = rxBuffer.split(b"\n")
    rxBuffer[:] = tail                     # keep any partial line for the next call
    latest = None
    for line in lines:
        m = _TELEMETRY.search(line)
        if m:
            latest = (int(m[1]), int(m[2]), int(m[3]))
    return latest


# ============================================================ process
def applyDeadzone(x):
    return 0.0 if abs(x) < _DEADZONE else x


def axesToCommand(rotAxis, linearAxis):
    linear = int(applyDeadzone(linearAxis) * _MAX_COMMAND)    # up -> forwards
    rot    = int(-applyDeadzone(rotAxis) * _MAX_COMMAND)   # right -> clockwise -> negative rot
    return linear, rot


def wheelSpeeds(prev, curr):
    """Wheel surface speeds (FR, FL) in m/s between two telemetry samples."""
    dt = ((curr[2] - prev[2]) % 2**32) * 1e-6        # micros() wraps at 2^32
    if dt == 0:
        return 0.0, 0.0
    k = 2 * math.pi * _WHEEL_RADIUS_M / (_TICKS_PER_REV * dt)
    return (curr[0] - prev[0]) * k, (curr[1] - prev[1]) * k


# ============================================================ actuate
def writeCommand(arduino, linear, rot):
    arduino.write(f"linear:{linear} rot:{rot}\n".encode())


# ============================================================ main
def main():
    arduino  = _connectArduino(_BAUD_RATE, _CONNECTION_RETRY_ATTEMPTS)
    joystick = F710Controller(maxRetryAttempts=_CONNECTION_RETRY_ATTEMPTS)
    joystick.connect()

    rxBuffer   = bytearray()
    prevSample = None
    vFR = vFL  = 0.0
    nextTick   = time.monotonic()

    try:
        while True:
            rightX, rightY = joystick.getNormalisedAxes(joystickType = "right")
            leftX, leftY   = joystick.getNormalisedAxes(joystickType = "left")

            sample = readEncoders(arduino, rxBuffer)

            linear, rot = axesToCommand(
                rotAxis    = leftX,
                linearAxis = rightY
            )
            if sample is not None:
                if prevSample is not None:
                    vFR, vFL = wheelSpeeds(prevSample, sample)
                prevSample = sample

            writeCommand(arduino, linear, rot)

            fr, fl = (prevSample[0], prevSample[1]) if prevSample else (0, 0)
            print(f"linear:{linear:+4d} rot:{rot:+4d} | FR:{fr:+8d} FL:{fl:+8d} | "
                  f"vFR:{vFR:+.3f} vFL:{vFL:+.3f} m/s")

            nextTick += _LOOP_PERIOD_S
            time.sleep(max(0.0, nextTick - time.monotonic()))
    except KeyboardInterrupt:
        print("\n! Stopping")
    finally:
        try:
            writeCommand(arduino, 0, 0)
        except serial.SerialException:
            pass
        arduino.close()


if __name__ == "__main__":
    main()