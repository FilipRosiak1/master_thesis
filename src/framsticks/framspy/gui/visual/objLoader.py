from gui.visual.loader import Loader
from gui.visual.modelData import ModelData
from gui.visual.modelTexture import ModelTexture
import glm

class OBJLoader:
    class PackedVertex:
        def __init__(self, position, uv, normal) -> None:
            self.position = position
            self.uv = uv
            self.normal = normal

    @staticmethod
    def loadOBJ(objFileName: str, textureFileName: str, loader: Loader):
        vertices, uvs, normals = OBJLoader._loadOBJ(objFileName)
        indices, indexed_vertices, indexed_uvs, indexed_normals = OBJLoader._indexVBO(vertices, uvs, normals)
        verticesf = OBJLoader._vec3ToFloat(indexed_vertices)
        uvsf = OBJLoader._vec2ToFloat(indexed_uvs)
        normalsf = OBJLoader._vec3ToFloat(indexed_normals)
        rawModel = loader.loadToVao5(verticesf, uvsf, normalsf, indices, len(vertices))
        model = ModelData(rawModel, ModelTexture(loader.loadTexture(textureFileName) if textureFileName else 0))
        return model

    @staticmethod
    def _loadOBJ(path):
        vertexIndices = []
        uvIndices = []
        normalIndices = []
        temp_vertices = []
        temp_uvs = []
        temp_normals = []
        out_vertices = []
        out_uvs = []
        out_normals = []

        with open(path, "r") as file:
            while (line := file.readline()):
                if line[0:2] == 'v ':
                    currentLine = line.split()
                    vertex = glm.vec3(float(currentLine[1]), float(currentLine[2]), float(currentLine[3]))
                    temp_vertices.append(vertex)
                elif line[0:2] == 'vt':
                    currentLine = line.split()
                    uv = glm.vec2(float(currentLine[1]), float(currentLine[2]))
                    temp_uvs.append(uv)
                elif line[0:2] == 'vn':
                    currentLine = line.split()
                    normal = glm.vec3(float(currentLine[1]), float(currentLine[2]), float(currentLine[3]))
                    temp_normals.append(normal)
                elif line[0] == 'f':
                    currentLine = line.split()
                    vertexIndex = [0,0,0]
                    uvIndex = [0,0,0]
                    normalIndex = [0,0,0]
                    for i in range(1, 4):
                        trip = currentLine[i].split('/')
                        vertexIndex[i-1] = int(trip[0])
                        if len(trip) > 1:
                            if trip[1]:
                                uvIndex[i-1] = int(trip[1])
                            else:
                                uvIndex[i-1] = 0
                        if len(trip) > 2:
                            normalIndex[i-1] = int(trip[2])
                    vertexIndices.extend(vertexIndex)
                    uvIndices.extend(uvIndex)
                    normalIndices.extend(normalIndex)
            
            for i in range(len(vertexIndices)):
                vertex = temp_vertices[vertexIndices[i] - 1]
                out_vertices.append(vertex)
                if temp_uvs:
                    uv = temp_uvs[uvIndices[i] - 1]
                else:
                    uv = glm.vec2(0)
                out_uvs.append(uv)
                if temp_normals:
                    normal = temp_normals[normalIndices[i] - 1]
                else:
                    normal = glm.vec3(0)
                out_normals.append(normal)

        return out_vertices, out_uvs, out_normals

    @staticmethod
    def _indexVBO(vertices, uvs, normals):
        out_indices = []
        out_vertices = []
        out_uvs = []
        out_normals = []
        vertexToOutIndex = {}

        for vertex, uv, normal in zip(vertices, uvs, normals):
            packed = OBJLoader.PackedVertex(vertex, uv, normal)
            found, index = OBJLoader._getSimilarVertexIndex(packed, vertexToOutIndex)

            if not found:
                out_vertices.append(vertex)
                out_uvs.append(uv)
                out_normals.append(normal)
                index = len(out_vertices) - 1
                vertexToOutIndex[packed] = index
            
            out_indices.append(index)
        
        return out_indices, out_vertices, out_uvs, out_normals

    @staticmethod
    def _getSimilarVertexIndex(vertex, VertexToOutIndex: dict):
        result = VertexToOutIndex.get(vertex)
        return True if result else False, result

    @staticmethod
    def _vec3ToFloat(vector):
        out = []
        for v in vector:
            out.append(v.x)
            out.append(v.y)
            out.append(v.z)
        return out

    @staticmethod
    def _vec2ToFloat(vector):
        out = []
        for v in vector:
            out.append(v.x)
            out.append(v.y)
        return out