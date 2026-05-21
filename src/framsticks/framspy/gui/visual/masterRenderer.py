from gui.visual.player import Player
from gui.visual.entity import Entity
import glm
import OpenGL.GL as gl
from gui.visual.camera import Camera
from gui.visual.staticShader import StaticShader
from gui.visual.entityRenderer import EntityRenderer
from gui.visual.skyboxRenderer import SkyboxRenderer
from gui.visual.worldRenderer import WorldRenderer

class MasterRenderer:
    FOV = 70
    NEAR_PLANE = 0.1
    FAR_PLANE = 17500 # max world size + some offset
    INITIAL_WIDTH = 800
    INITIAL_HEIGHT = 600

    def __init__(self, loader) -> None:
        self._width = self.INITIAL_WIDTH
        self._height = self.INITIAL_HEIGHT
        self._createProjectionMatrix()
        self._shader = StaticShader()
        self._renderer = EntityRenderer(self._shader, self.projectionMatrix)
        self._skyboxRenderer = SkyboxRenderer(loader, self.projectionMatrix)
        self.worldRenderer = WorldRenderer(loader, self.projectionMatrix, self._shader)
        self._entities = {}
        self._isTexturesOn = True

    def renderScene(self, player: Player, entities: list, camera: Camera):
        for entity in entities:
            self.processEntity(entity)

        self.render(camera)

    def render(self, camera: Camera):
        self.__prepare()
        self._shader.start()
        self._shader.loadViewMatrix(camera)
        self._renderer.render(self._entities, self._isTexturesOn)
        self.worldRenderer.render(camera)
        self._skyboxRenderer.render(camera)
        self._shader.stop()
        self._entities.clear()

    def processEntity(self, entity: Entity):
        entityModel = entity.modelData
        batch = self._entities.get(entityModel)
        if batch:
            batch.append(entity)
        else:
            newBatch = [entity]
            self._entities[entityModel] = newBatch

    def toggleTextures(self):
        self._isTexturesOn = not self.isTexturesOn

    def cleanUp(self):
        self._shader.cleanUp()

    def resize(self, width, height):
        self._width = width
        self._height = height
        self._createProjectionMatrix()
        self._shader.start()
        self._renderer.loadProjectionMatrix(self.projectionMatrix)
        self._shader.stop()
        self._skyboxRenderer.shader.start()
        self._skyboxRenderer.loadProjectionMatrix(self.projectionMatrix)
        self._skyboxRenderer.shader.stop()
        self.worldRenderer.shader.start()
        self.worldRenderer.loadProjectionMatrix(self.projectionMatrix)
        self.worldRenderer.shader.stop()

    def __prepare(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1)

    def _createProjectionMatrix(self):
        self.projectionMatrix = glm.perspective(self.FOV, self._width / self._height, self.NEAR_PLANE, self.FAR_PLANE)

    def reloadWorld(self, loader, worldType: int, simType: int, worldSize: float, worldMap: str, worldBoundaries: int, worldWaterLevel: float) -> None:
        self.worldRenderer.reloadWorld(loader, worldType, simType, worldSize, worldMap, worldBoundaries, worldWaterLevel)