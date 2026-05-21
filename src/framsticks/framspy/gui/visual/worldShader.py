from gui.visual.shaderProgram import ShaderProgram
import OpenGL.GL as gl
import glm

class WorldShader(ShaderProgram):
    def __init__(self) -> None:
        super().__init__("visual/shaders/WorldVertexShader.glsl", "visual/shaders/WorldFragmentShader.glsl", "visual/shaders/WorldGeometryShader.glsl")
        self.bindAttributes()
        gl.glLinkProgram(self.programID)
        gl.glValidateProgram(self.programID)
        self.getAllUniformLocations()

    def loadProjectionMatrix(self, matrix):
        self.loadMatrix4(self.location_projectionMatrix, matrix)

    def loadViewMatrix(self, camera):
        viewMatrix = glm.lookAt(camera.position, camera.player.position, glm.vec3(0, 0, 1))
        self.loadMatrix4(self.location_viewMatrix, viewMatrix)

    def loadMode(self, mode: int):
        self.loadInt(self.location_mode, mode)

    def loadWorldSize(self, size: float):
        self.loadFloat(self.location_worldSize, size)

    def loadWaterLevel(self, level: float):
        self.loadFloat(self.location_waterLevel, level)

    def bindAttributes(self):
        self.bindAttribute(0, "position")
        self.bindAttribute(1, "color")

    def getAllUniformLocations(self):
        self.location_projectionMatrix = self.getUniformLocation("projectionMatrix")
        self.location_viewMatrix = self.getUniformLocation("viewMatrix")
        self.location_mode = self.getUniformLocation("mode")
        self.location_worldSize = self.getUniformLocation("worldSize")
        self.location_waterLevel = self.getUniformLocation("waterLevel")