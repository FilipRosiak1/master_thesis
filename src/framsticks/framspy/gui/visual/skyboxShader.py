from gui.visual.shaderProgram import ShaderProgram
import OpenGL.GL as gl
import glm

class SkyboxShader(ShaderProgram):
    def __init__(self) -> None:
        super().__init__("visual/shaders/SkyboxVertexShader.glsl", "visual/shaders/SkyboxFragmentShader.glsl")
        self.bindAttributes()
        gl.glLinkProgram(self.programID)
        gl.glValidateProgram(self.programID)
        self.getAllUniformLocations()

    def loadProjectionMatrix(self, matrix):
        self.loadMatrix4(self.location_projectionMatrix, matrix)

    def loadViewMatrix(self, camera):
        viewMatrix = glm.lookAt(camera.position, camera.player.position, glm.vec3(0, 0, 1))
        viewMatrix[3][0] = 0
        viewMatrix[3][1] = 0
        viewMatrix[3][2] = 0
        self.loadMatrix4(self.location_viewMatrix, viewMatrix)

    def bindAttributes(self):
        self.bindAttribute(0, "position")
        self.bindAttribute(1, "textureCoordinates")

    def getAllUniformLocations(self):
        self.location_projectionMatrix = self.getUniformLocation("projectionMatrix")
        self.location_viewMatrix = self.getUniformLocation("viewMatrix")
        location_modelTexture = self.getUniformLocation("modelTexture")