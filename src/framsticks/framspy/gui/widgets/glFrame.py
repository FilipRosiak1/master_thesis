from xmlrpc.client import Boolean
from gui.visual.entity import Entity
import OpenGL.GL as gl
from pyopengltk import OpenGLFrame
from gui.visual.camera import Camera
from gui.visual.player import Player
from gui.visual.masterRenderer import MasterRenderer
from gui.visual.loader import Loader
from gui.visual.objLoader import OBJLoader
from gui.visual.player import Player
from gui.visual.mouse import Mouse
from gui.framsutils.creature import Creature
from typing import List, Callable
import threading
import time as timetime
from gui.utils import Swap
from typing import Dict
from gui.visual.modelData import ModelData

class AppOgl(OpenGLFrame):
    class GLContext(object):
        pass

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.REFRESH_RATE = 0.1
        self.player = Player(10, 10, 0, 0, 0, 0)
        self.camera = Camera(self.player)
        self.entities = []
        self.frams_readCreatures: Callable[[None], List[Creature]] = None
        self.frams_readCreatures_color: Boolean = False
        self.onDraw: Callable[[], None] = lambda: None
        self.commThread = threading.Thread(target=self.commThread, daemon=True)
        self.commThreadLock = threading.Lock()
        self.read_creature_semaphore = threading.Semaphore()
        self.creatures: List[Creature] = []
        self.swap_buffer = Swap(init=[])
        self.swap_buffer.update([])
        self.swap_buffer.update([])
        self.commThread.start()

    #show tri-colored axis instead of balls
    SHOW_AXIS = False

    NEURON_SCALE = 1.5
    OBJ_PATH = "gui/res/obj/"
    TEX_PATH = "gui/res/img/"

    NEURONS = ["|", "@", "G", "Gpart", "T", "S"]

    def neuronname_in_filename(self, neuronname):
        if neuronname == "|":
            return "bend"
        elif neuronname == "@":
            return "rot"
        return neuronname

    def initgl(self):
        gl.glClearColor(0, 0, 0.4, 0)
        
        self.loader = Loader()

        if self.SHOW_AXIS:
            self.bodyPartModel = OBJLoader.loadOBJ(self.OBJ_PATH+"axis.obj", self.TEX_PATH+"axis.png", self.loader)
        else:
            self.bodyPartModel = OBJLoader.loadOBJ(self.OBJ_PATH+"part.obj", self.TEX_PATH+"green2.png", self.loader)
        self.jointModel = OBJLoader.loadOBJ(self.OBJ_PATH+"stick5red3.obj", self.TEX_PATH+"green2.png", self.loader)
        self.neurons: Dict[str, ModelData] = dict()
        for n in self.NEURONS:
            self.neurons[n] = OBJLoader.loadOBJ(self.OBJ_PATH+"neuro-"+self.neuronname_in_filename(n)+".obj", self.TEX_PATH+"amber.jpg", self.loader)

        self.renderer = MasterRenderer(self.loader)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearDepth(1.0)
        gl.glEnable(gl.GL_TEXTURE_2D)

        self.initalized = True
        self._resize(self.width, self.height)

    def init_context(self):
        self.context = self.GLContext()

    def redraw(self):
        #call onDraw callback  only for frams library 
        self.onDraw()
        self.entities.clear()
        self.creatures: List[Creature] = self.swap_buffer.get()
        #for every creature, prepare entities for every parts, joints and neurons
        for creature in self.creatures:
            parts = creature.mechParts
            joints = creature.joints
            neurons = creature.neurons

            '''
            TODO: solid shapes should be handled here
            obj models should be loaded in initgl method and used depending on the shape type (probably need to add new lists to Creature class).
            Position should already be loaded in the part object, as well as rotation matrix, you need to add scale in FramsSocket and FramsLib.
            You can add scale in each axis in the Entity constructor or as independent properties of the Entity class.
            scaleX, scaleY, scaleZ have higher priority than scale only if all three (x, y, z) are set.
            '''
            for i, part in enumerate(parts):
                rotMatrix = creature.partRotationMatrix(i)
                partEntity = Entity(part[0], part[1], part[2], self.bodyPartModel, scale=0.7, rotMatrix=rotMatrix)
                if i < len(creature.colorsPart):
                    partEntity.color = creature.colorsPart[i]
                self.entities.append(partEntity)

            for i in range(len(joints)):
                translation = creature.jointTranslation(i)
                rotMatrix = creature.jointRotation(i)
                jointEntity = Entity(translation[0], translation[1], translation[2], self.jointModel, rotMatrix=rotMatrix)
                jointEntity.scaleX = creature.jointLength(i)
                jointEntity.scaleY = 1
                jointEntity.scaleZ = 1
                if i < len(creature.colorsJoint):
                    partEntity.color = creature.colorsJoint[i]
                self.entities.append(jointEntity)

            for i, neuron in enumerate(neurons):
                if creature.styleNeuron[i] in self.neurons:
                    style = creature.styleNeuron[i]
                    if neuron[0] >= 0:
                        translation = creature.partTranslation(neuron[0])
                        rotMatrix = creature.partRotationMatrix(neuron[0])
                        rotMatrix = rotMatrix * creature.neuronRelativeOrient[i]
                        neuronEntity = Entity(translation[0], translation[1], translation[2], self.neurons[style], scale=self.NEURON_SCALE, rotMatrix=rotMatrix)
                    else:
                        translation = creature.jointTranslation(neuron[1])
                        if style == 'G':
                            rotMatrix = creature.jointRotation(neuron[1])
                        else:
                            rotMatrix = creature.partRotationMatrix(joints[neuron[1]][0])
                        rotMatrix = rotMatrix * creature.neuronRelativeOrient[i]
                        neuronEntity = Entity(translation[0], translation[1], translation[2], self.neurons[style], scale=self.NEURON_SCALE, rotMatrix=rotMatrix)
                    self.entities.append(neuronEntity)

        self.camera.move()
        self.renderer._isTexturesOn = not self.frams_readCreatures_color
        self.renderer.renderScene(self.player, self.entities, self.camera)

    #reload all creatures from frams with self.REFRESH_RATE interval
    def commThread(self):
        while True:
            if self.frams_readCreatures:
                with self.read_creature_semaphore:
                    c = self.frams_readCreatures(self.frams_readCreatures_color)
                self.swap_buffer.update(c)
            timetime.sleep(self.REFRESH_RATE)

    def onResize(self, event):
        width, height = event.width, event.height
        self._resize(width, height)

    def _resize(self, width, height):
        self.width = width
        self.height = height
        try:
            self.initalized
            gl.glViewport(0, 0, width, height)
            self.renderer.resize(width, height)
        except AttributeError:
            pass

    def onMouseMotion(self, event):
        Mouse.setXY(event.x, event.y)

    def onScroll(self, event):
        if event.delta > 0:
            Mouse.incrementDWheel(1)
        else:
            Mouse.incrementDWheel(-1)

    def onMouseClick(self, event):
        if event.num == 1:
            Mouse.setButton(Mouse.MOUSE_LEFT)
        elif event.num == 3:
            Mouse.setButton(Mouse.MOUSE_RIGHT)

    def onMouseRelease(self, event):
        Mouse.setButton(Mouse.MOUSE_NONE)

    def onMouseEnter(self, event):
        Mouse.setXY(event.x, event.y)

    def reloadWorld(self, worldType: int, simType: int, worldSize: float, worldMap: str, worldBoundaries: int, worldWaterLevel: float) -> None:
        self.renderer.reloadWorld(self.loader, worldType, simType, worldSize, worldMap, worldBoundaries, worldWaterLevel)
        self.camera.setPosition(worldSize / 2.0, worldSize / 2.0)
        self.camera.setZoomScale(worldSize / 20)
        self.camera.setZoom(worldSize)