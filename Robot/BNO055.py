import board
import busio

class BNO055:
    def __init__(self, deviceAddress, registerAddress):
        self._deviceAddress   = deviceAddress
        self._buffer          = bytearray(1)
        self._i2c             = busio.I2C(
            scl = board.SCL,
            sda = board.SDA,
        )

    def read(self) -> float:
        self._i2c.readfrom_into(
            address = self._deviceAddress,
            buffer  = self._buffer,
        )
        self._i2c.unlock()

        return self._buffer[0]

