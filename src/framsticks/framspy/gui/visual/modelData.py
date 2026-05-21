from gui.visual.rawModel import RawModel
from gui.visual.modelTexture import ModelTexture

class ModelData:
    def __init__(self, rawModel: RawModel, modelTexture: ModelTexture) -> None:
        self.rawModel = rawModel
        self.texture = modelTexture