import select
import time
import evdev
import numpy


class F710Controller:
    LEFT_X  = evdev.ecodes.ABS_X
    LEFT_Y  = evdev.ecodes.ABS_Y
    RIGHT_X = evdev.ecodes.ABS_RX
    RIGHT_Y = evdev.ecodes.ABS_RY
    MAX_AXIS_VALUE = 32767

    def __init__(self, maxRetryAttempts, targetName = "Logitech"):
        self._targetName       = targetName
        self._device           = None
        self._maxRetryAttempts = maxRetryAttempts

    def connect(self):
        print(f"! Searching for {self._targetName} device")
        for _ in range(self._maxRetryAttempts):
            for p in evdev.list_devices():
                possibleDevice = evdev.InputDevice(p)
                if self._targetName in possibleDevice.name:
                    self._device = possibleDevice
                    print(f"    - Connected to {self._targetName} device on {self._device.path}")
                    return self._device
                possibleDevice.close()
            print(f"    - Cannot find {self._targetName} device. Retrying")
            time.sleep(2)
        raise RuntimeError(f"! ERROR: could not connect to {self._targetName} device after {self._maxRetryAttempts} attempts")

    def isConnected(self) -> bool:
        if self._device is not None and self._device.path in evdev.list_devices():
            return True
        print(f"! FAULT: cannot locate {self._targetName} device")
        return False

    def getNormalisedAxes(self, joystickType:str):
        def getAxisValue(axis) -> float:
            return numpy.clip(
                self._device.absinfo(axis).value / 32767,
                a_min = -1.0,
                a_max = 1.0
            )
        """
        Returns 2 axes of the joystick they are associated to. The tuple will always be in the (pX, pY) format. Both elements of the tuple
        are normalised by the F710's maximum axis value of 32767.
        """
        if joystickType == "left":
            return getAxisValue(self.LEFT_X), -getAxisValue(self.LEFT_Y)
        elif joystickType == "right":
            return getAxisValue(self.RIGHT_X), -getAxisValue(self.RIGHT_Y)

    def pollButtonPresses(self) -> list:
        pressed = []
        r, _, _ = select.select(
            [self._device.fd],
            [],
            [],
            0
        )
        if r:
            for e in self._device.read():
                if e.type == evdev.ecodes.EV_KEY and e.value == 1:
                    pressed.append(e.code)
        return pressed

if __name__ == "__main__":
    controller = F710Controller(maxRetryAttempts = 5)
    controller.connect()

    while controller.isConnected():
        leftX, leftY = controller.getNormalisedAxes("left")
        rightX, rightY = controller.getNormalisedAxes("right")

        print(
            f"leftX:{leftX}, leftY:{leftY}, rightX:{rightX}, rightY:{rightY}"
        )
        time.sleep(0.1)


