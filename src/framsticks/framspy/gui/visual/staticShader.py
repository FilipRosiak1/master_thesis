from gui.visual.shaderProgram import ShaderProgram
from gui.visual.camera import Camera
import OpenGL.GL as gl
import glm

class StaticShader(ShaderProgram):
    def __init__(self) -> None:
        super().__init__("visual/shaders/vertexShader.glsl", "visual/shaders/fragmentShader.glsl")

        self.location_transformationMatrix = 0
        self.location_projectionMatrix = 0
        self.location_viewMatrix = 0
        self.location_modelTexture = 0

        self.bindAttributes()
        gl.glLinkProgram(self.programID)
        gl.glValidateProgram(self.programID)
        self.getAllUniformLocations()

    def loadProjectionMatrix(self, projection):
        self.loadMatrix4(self.location_projectionMatrix, projection)

    def loadViewMatrix(self, camera: Camera):
        viewMatrix = glm.lookAt(camera.position, camera.player.position, glm.vec3(0, 0, 1))
        self.loadMatrix4(self.location_viewMatrix, viewMatrix)

    def loadTransformationMatrix(self, matrix):
        self.loadMatrix4(self.location_transformationMatrix, matrix)

    def loadColor(self, color: glm.vec3):
        self.loadVector3(self.loaction_color, color)

    def loadTextureOn(self, textureOn: bool):
        self.loadFloat(self.location_textureOn, 1.0 if textureOn else 0.0)

    def bindAttributes(self):
        self.bindAttribute(0, "position")
        self.bindAttribute(1, "textureCoordinates")
        self.bindAttribute(2, "normal")

    def getAllUniformLocations(self):
        self.location_transformationMatrix = self.getUniformLocation("transformationMatrix")
        self.location_projectionMatrix = self.getUniformLocation("projectionMatrix")
        self.location_viewMatrix = self.getUniformLocation("viewMatrix")
        self.location_modelTexture = self.getUniformLocation("modelTexture")
        self.loaction_color = self.getUniformLocation("color")
        self.location_textureOn = self.getUniformLocation("textureOn")