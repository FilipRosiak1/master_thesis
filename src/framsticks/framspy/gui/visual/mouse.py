class Mouse:
    MOUSE_NONE = 0
    MOUSE_LEFT = 1
    MOUSE_RIGHT = 2

    _dwheel = 0
    _dbutton = MOUSE_NONE
    _xactual = -1
    _yactual = -1
    _dx = -1
    _dy = -1

    @staticmethod
    def setDWheel(value: float):
        Mouse._dwheel = value

    @staticmethod
    def incrementDWheel(value: float):
        Mouse._dwheel += value

    @staticmethod
    def setButton(button: int):
        Mouse._dbutton = button

    @staticmethod
    def setXY(x: int, y: int):
        Mouse._xactual = x
        Mouse._yactual = y

    @staticmethod
    def getDWheel() -> float:
        t = Mouse._dwheel
        Mouse._dwheel = 0
        return t

    @staticmethod
    def getX() -> float:
        return Mouse._xactual

    @staticmethod
    def getY() -> float:
        return Mouse._yactual

    @staticmethod
    def isButtonDown(button: int) -> bool:
        return Mouse._dbutton == button