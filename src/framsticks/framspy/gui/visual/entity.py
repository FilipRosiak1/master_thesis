import glm
from gui.visual.modelData import ModelData
from OpenGL.GL import *

class Entity():
    def __init__(self, x: float, y: float, z: float, model: ModelData, scale: float = None, rotX: float = None, rotY: float = None, rotZ: float = None, angle: float = None, vector: glm.vec3 = None, color: glm.vec3 = None, rotMatrix: glm.mat4 = None):
        self.position = glm.vec3(x, y, z)
        self.rotX = rotX
        self.rotY = rotY
        self.rotZ = rotZ
        self.rotMatrix = rotMatrix
        self.angle = angle
        self.vector = vector
        self.scale = scale
        self.modelData = model
        self.textureIndex = 0
        self.scaleX = None
        self.scaleY = None
        self.scaleZ = None
        self.color = color

    def getTextureXOffset(self):
        column = self.textureIndex % self.modelData.texture.numberOfRows
        return column / self.modelData.texture.numberOfRows

    def getTextureYOffset(self):
        row = self.textureIndex % self.modelData.texture.numberOfRows
        return row / self.modelData.texture.numberOfRows

    def increasePosition(self, dx: float, dy: float, dz: float):
        self.position += glm.vec3(dx, dy, dz)

    def increaseRotation(self, dx: float, dy: float, dz: float):
        self.rotX += dx
        self.rotY += dy
        self.rotZ += dz