from gui.visual.loader import Loader
from gui.visual.worldShader import WorldShader
from gui.visual.staticShader import StaticShader
from gui.visual.camera import Camera
from gui.visual.objLoader import OBJLoader
import OpenGL.GL as gl
from typing import Tuple, List
import numpy as np
import re
import glm

class WorldRenderer:
    WORLD_Z_OFFSET = -0.2

    def __init__(self, loader: Loader, projectionMatrix, staticShader: StaticShader) -> None:
        self.shader = WorldShader()
        self._createWater(loader)
        self.staticShader: StaticShader = staticShader

        self.fence = OBJLoader.loadOBJ("gui/res/obj/fence-element.obj", "gui/res/img/wood.png", loader)
        self.teleport = OBJLoader.loadOBJ("gui/res/obj/teleport.obj", "gui/res/img/teleportglow.png", loader)

        self.boundary: List[Tuple[glm.vec3, float]] = []
        self.worldType = 0
        self.worldSize = 20
        self.worldBoundaries = 0
        self.z_offset = 0

        self.shader.start()
        self.loadProjectionMatrix(projectionMatrix)
        self.shader.stop()

    def render(self, camera: Camera):
        if hasattr(self, "plane"):
            self.shader.start()
            self.shader.loadViewMatrix(camera)
            self.shader.loadMode(0)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            #render world plane
            gl.glBindVertexArray(self.plane.vaoID)
            gl.glEnableVertexAttribArray(0)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.plane.vertexCount))
            gl.glDisableVertexAttribArray(0)
            gl.glBindVertexArray(0)

            #render water plane
            gl.glBindVertexArray(self.water.vaoID)
            gl.glEnableVertexAttribArray(0)
            gl.glEnableVertexAttribArray(1)
            self.shader.loadMode(1)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.water.vertexCount))
            gl.glDisableVertexAttribArray(0)
            gl.glBindVertexArray(0)
            gl.glBindVertexArray(1)

            gl.glDisable(gl.GL_BLEND)

            #render bounds
            self.renderBounds()
            
            self.shader.stop()

    def loadProjectionMatrix(self, projectionMatrix):
        self.shader.loadProjectionMatrix(projectionMatrix)

    def reloadWorld(self, loader: Loader, worldType: int, simType: int, worldSize: float, worldMap: str, worldBoundaries: int, worldWaterLevel: float):
        self.worldType = worldType
        self.worldSize = worldSize
        self.worldBoundaries = worldBoundaries

        if simType == 0:
            self.z_offset = self.WORLD_Z_OFFSET
        else:
            self.z_offset = 0

        v = self._loadQuads(worldMap)
        self.plane = loader.loadToVao1(v, len(v) / 3)
        self.shader.start()
        self.shader.loadWorldSize(worldSize)
        self.shader.loadWaterLevel(worldWaterLevel + self.z_offset)
        self.shader.stop()

        if worldBoundaries == 1:
            self._addFence()
        elif worldBoundaries == 2:
            self._addTeleport()

    def _createBlock(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], 
                            p3: Tuple[float, float, float], p4: Tuple[float, float, float]) -> List[float]:
        vertices = []
        vertices.append(p1[0])
        vertices.append(p1[1])
        vertices.append(p1[2])

        vertices.append(p2[0])
        vertices.append(p2[1])
        vertices.append(p2[2])

        vertices.append(p3[0])
        vertices.append(p3[1])
        vertices.append(p3[2])

        vertices.append(p1[0])
        vertices.append(p1[1])
        vertices.append(p1[2])

        vertices.append(p3[0])
        vertices.append(p3[1])
        vertices.append(p3[2])

        vertices.append(p4[0])
        vertices.append(p4[1])
        vertices.append(p4[2])

        return vertices

    def _createColorsForBlock(self, color: Tuple[float, float, float]) -> List[float]:
        return [color[0], color[1], color[2]] * 6

    def _createWater(self, loader: Loader):
        vertices = self._createBlock((0,0,-0.01), (1,0,-0.01), (1,1,-0.01), (0,1,-0.01))
        colors = self._createColorsForBlock((0,0,1))

        self.water = loader.loadToVao2(vertices, colors, len(vertices) / 3)

    def _loadQuads(self, a: str):
        points = []
        indices = []
        lines = a.splitlines()
        self.primitive = None

        for line in lines:
            if line[0] == 'v':
                v = line.split()
                points.append((float(v[1]), float(v[2]), float(v[3]) + self.z_offset))
            elif line[0] == 'f':
                f = line.split()
                if len(f) == 4:
                    indices.append(int(f[1]))
                    indices.append(int(f[2]))
                    indices.append(int(f[3]))
                elif len(f) == 5:
                    indices.append(int(f[1]))
                    indices.append(int(f[2]))
                    indices.append(int(f[3]))

                    indices.append(int(f[1]))
                    indices.append(int(f[3]))
                    indices.append(int(f[4]))

        vertices = []
        for i in indices:
            vertices.append(points[i-1][0])
            vertices.append(points[i-1][1])
            vertices.append(points[i-1][2])

        return vertices

    def renderBounds(self):
        if self.worldBoundaries == 1:
            self.staticShader.start()
            gl.glBindVertexArray(self.fence.rawModel.vaoID)
            gl.glEnableVertexAttribArray(0)
            gl.glEnableVertexAttribArray(1)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.fence.texture.id)
            self.staticShader.loadTextureOn(True)

            for b in self.boundary:
                matrix = glm.mat4(1)
                matrix = glm.translate(matrix, b[0])
                matrix = glm.rotate(matrix, 3.14/2, glm.vec3(1,0,0))
                matrix = glm.rotate(matrix, b[1], glm.vec3(0,1,0))
                self.staticShader.loadTransformationMatrix(matrix)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.fence.rawModel.vertexCount))

            self.staticShader.stop()
        elif self.worldBoundaries == 2:
            self.staticShader.start()
            gl.glBindVertexArray(self.teleport.rawModel.vaoID)
            gl.glEnableVertexAttribArray(0)
            gl.glEnableVertexAttribArray(1)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.teleport.texture.id)

            for b in self.boundary:
                matrix = glm.mat4(1)
                matrix = glm.translate(matrix, b[0])
                matrix = glm.rotate(matrix, 3.14/2, glm.vec3(1,0,0))
                matrix = glm.rotate(matrix, b[1], glm.vec3(0,1,0))
                self.staticShader.loadTransformationMatrix(matrix)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.teleport.rawModel.vertexCount))

            self.staticShader.stop()

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glBindVertexArray(0)

    def _addFence(self):
        i = 0
        self.boundary.clear()
        while i < self.worldSize / 2:
            self.boundary.append((glm.vec3(i * 2, 0, self.z_offset), 0))
            self.boundary.append((glm.vec3(i * 2, self.worldSize, self.z_offset), 0))
            self.boundary.append((glm.vec3(0, i * 2, self.z_offset), glm.radians(90)))
            self.boundary.append((glm.vec3(self.worldSize, i * 2, self.z_offset), glm.radians(90)))
            i += 1

    def _addTeleport(self):
        i = 0
        self.boundary.clear()
        while i < self.worldSize / 3:
            self.boundary.append((glm.vec3(i * 3, 0, self.z_offset), 0))
            self.boundary.append((glm.vec3(i * 3, self.worldSize, self.z_offset), 0))
            self.boundary.append((glm.vec3(0, i * 3, self.z_offset), glm.radians(-90)))
            self.boundary.append((glm.vec3(self.worldSize, i * 3, self.z_offset), glm.radians(-90)))
            i += 1
