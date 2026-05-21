from gui.visual.loader import Loader
from gui.visual.camera import Camera
from gui.visual.skyboxShader import SkyboxShader
import glm
import OpenGL.GL as gl

class SkyboxRenderer:
    def __init__(self, loader: Loader, projectionMatrix) -> None:
        self.shader = SkyboxShader()
        vertices, text = self.createCube()
        self.skycube = loader.loadToVao3(vertices, text, int(len(vertices)/3))
        self.texture = loader.loadCubemap([
            "gui/res/img/field3.png",
            "gui/res/img/field5.png",
            "gui/res/img/field4.png",
            "gui/res/img/field2.png",
            "gui/res/img/field1.png",
            "gui/res/img/field6.png",
        ])
        self.shader.start()
        self.loadProjectionMatrix(projectionMatrix)
        self.shader.stop()

    def loadProjectionMatrix(self, projectionMatrix):
        self.shader.loadProjectionMatrix(projectionMatrix)

    def render(self, camera: Camera):
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.shader.start()
        self.shader.loadViewMatrix(camera)
        gl.glBindVertexArray(self.skycube.vaoID)
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        self.bindTextures()
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.skycube.vertexCount)
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glBindVertexArray(0)
        self.shader.stop()
        gl.glDepthFunc(gl.GL_LESS)

    def bindTextures(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texture)

    def createCube(self):
        skyboxVertices = [       
        -1.0,  -1.0, 1.0,
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0,  -1.0, 1.0,
        -1.0,  -1.0, 1.0,

        -1.0, 1.0,  -1.0,
        -1.0, -1.0, -1.0,
        -1.0,  -1.0, 1.0,
        -1.0,  -1.0, 1.0,
        -1.0,  1.0,  1.0,
        -1.0, 1.0,  -1.0,

        1.0, -1.0, -1.0,
        1.0, 1.0,  -1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0, -1.0, 1.0,
        1.0, -1.0, -1.0,

        -1.0, 1.0,  -1.0,
        -1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0, 1.0,  -1.0,
        -1.0, 1.0,  -1.0,

        -1.0,  -1.0, 1.0,
        1.0,  -1.0, 1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  -1.0, 1.0,

        -1.0, -1.0, -1.0,
        -1.0, 1.0,  -1.0,
        1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        -1.0, 1.0,  -1.0,
        1.0, 1.0,  -1.0]

        return skyboxVertices, skyboxVertices