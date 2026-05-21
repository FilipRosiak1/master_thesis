import glm
from gui.visual.mouse import Mouse
from gui.visual.player import Player

class Camera:
    def __init__(self, player: Player) -> None:
        self.position = glm.vec3(0, 0, 1)
        self.pith = 10
        self.yaw = 0
        self.roll = 0
        self.angleAroundPlayer = 0
        self.player = player
        self.distanceFromPlayer = 10
        self.mx = 0
        self.my = 0
        self.scale = 1

    def setPosition(self, x, y):
        self.player.position = glm.vec3(x, y, 0)

    def setZoomScale(self, scale):
        self.scale = scale

    def setZoom(self, zoom):
        self.distanceFromPlayer = 0.70140280561122 * zoom - 0.028056112224449 #values from linear interpolation

    def move(self):
        self.calculateZoom()
        self.calculatePitchAndAngleAroundPLayer()
        horizontalDistance = self.calculateHorizotalDistance()
        verticalDistance = self.calculateVerticalDistance()
        self.calculateCameraPosition(horizontalDistance, verticalDistance)

    def calculateZoom(self):
        zoomLevel = Mouse.getDWheel() * self.scale
        self.distanceFromPlayer -= zoomLevel
        if self.distanceFromPlayer < 0.1:
            self.distanceFromPlayer = 0.1

    def calculateCameraPosition(self, horizontalDistance, verticalDistance):
        theta = self.player.rotY + self.angleAroundPlayer
        offsetX = horizontalDistance * glm.sin(glm.radians(theta))
        offsetY = horizontalDistance * glm.cos(glm.radians(theta))
        self.position.x = self.player.position.x - offsetX
        self.position.y = self.player.position.y + offsetY
        self.position.z = self.player.position.z + verticalDistance
        yaw = 180 - theta

    def calculateHorizotalDistance(self):
        return self.distanceFromPlayer * glm.cos(glm.radians(self.pith))

    def calculateVerticalDistance(self):
        return self.distanceFromPlayer * glm.sin(glm.radians(self.pith))

    def calculatePitchAndAngleAroundPLayer(self):
        if Mouse.isButtonDown(Mouse.MOUSE_LEFT):
            dx = Mouse.getX() - self.mx
            dy = Mouse.getY() - self.my
            pitchChange = dy * 0.1
            self.pith += pitchChange
            if self.pith < -90:
                self.pith = -89.99
            if self.pith > 90:
                self.pith = 89.99

            angleChange = dx * 0.3
            self.angleAroundPlayer -= angleChange
        elif Mouse.isButtonDown(Mouse.MOUSE_RIGHT):
            dx = Mouse.getX() - self.mx
            dy = Mouse.getY() - self.my
            theta = glm.radians(self.player.rotY + self.angleAroundPlayer)
            scale = self.distanceFromPlayer * 10 / 250
            self.player.position.x += (dx * glm.cos(theta) + dy * glm.sin(theta)) * 0.1 * scale
            self.player.position.y += (dx * glm.sin(theta) - dy * glm.cos(theta)) * 0.1 * scale
        self.mx = Mouse.getX()
        self.my = Mouse.getY()