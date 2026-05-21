from gui.visual.entity import Entity

class Player(Entity):
    def __init__(self, x, y, z, rotX, rotY, rotZ):
        super().__init__(x, y, z, 0, [], rotX, rotY, rotZ)