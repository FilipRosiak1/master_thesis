import OpenGL.GL as gl
from OpenGL.raw.GL.VERSION.GL_1_5 import GL_STATIC_DRAW
from gui.visual.rawModel import RawModel
from PIL import Image
import numpy as np
from typing import List

class Loader:
    def __init__(self) -> None:
        self.vaos = []
        self.vbos = []
        self.textures = []

    def loadToVao5(self, positions, textureCoords, normals, indieces, vertexCount: int):
        vaoid = self.createVAO()
        self.bindIndicesBuffer(indieces)
        self.storeDataInAttributeList(0, 3, positions)
        self.storeDataInAttributeList(1, 2, textureCoords)
        self.storeDataInAttributeList(2, 3, normals)
        self.unbindVAO()
        return RawModel(vaoid, vertexCount)

    def loadToVao3(self, positions, textureCoords, vertexCount: int):
        vaoid = self.createVAO()
        self.storeDataInAttributeList(0, 3, positions)
        self.storeDataInAttributeList(1, 2, textureCoords)
        self.unbindVAO()
        return RawModel(vaoid, vertexCount)

    def loadToVao2(self, positions, colors, vertexCount: int):
        vaoid = self.createVAO()
        self.storeDataInAttributeList(0, 3, positions)
        self.storeDataInAttributeList(1, 3, colors)
        self.unbindVAO()
        return RawModel(vaoid, vertexCount)

    def loadToVao1(self, positions, vertexCount: int):
        vaoid = self.createVAO()
        self.storeDataInAttributeList(0, 3, positions)
        self.unbindVAO()
        return RawModel(vaoid, vertexCount)

    def loadTexture(self, filename: str) -> int:
        image = Image.open(filename)
        img_data = np.array(list(image.getdata()), np.uint8)

        textureid = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, textureid)
        colors = gl.GL_RGB
        if image.mode == 'RGBA':
            colors = gl.GL_RGBA
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.width, image.height, 0, colors, gl.GL_UNSIGNED_BYTE, img_data)
        
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_LOD_BIAS, -2.4)

        self.textures.append(textureid)
        return textureid

    def loadCubemap(self, filenames: List[str]) -> int:
        if len(filenames) != 6:
            raise Exception("filenames must contain exactly 6 textures")
        textureid = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, textureid)

        for i, filename in enumerate(filenames):
            image = Image.open(filename)
            img_data = np.array(list(image.getdata()), np.uint8)
            colors = gl.GL_RGB
            if image.mode == "RGBA":
                colors = gl.GL_RGBA
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.GL_RGB, image.width, image.height, 0, colors, gl.GL_UNSIGNED_BYTE, img_data)

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        return textureid

    def cleanUp(self):
        for vao in self.vaos:
            gl.glDeleteVertexArrays(1, vao)

        for vbo in self.vbos:
            gl.glDeleteBuffers(1, vbo)

        for texture in self.textures:
            gl.glDeleteTextures(1, texture)

    def createVAO(self):
        vao_id = gl.glGenVertexArrays(1)
        self.vaos.append(vao_id)
        gl.glBindVertexArray(vao_id)
        return vao_id

    def storeDataInAttributeList(self, attributeNumber: int, coordinateSize: int, data: list):
        vboid = gl.glGenBuffers(1)
        self.vbos.append(vboid)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vboid)
        npdata = np.array(data, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(data) * 4, npdata, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(attributeNumber, coordinateSize, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def unbindVAO(self):
        gl.glBindVertexArray(0)

    def bindIndicesBuffer(self, indices):
        vboid = gl.glGenBuffers(1)
        self.vbos.append(vboid)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, vboid)
        npindices = np.array(indices, dtype=np.int32)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(indices) * 4, npindices, GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)