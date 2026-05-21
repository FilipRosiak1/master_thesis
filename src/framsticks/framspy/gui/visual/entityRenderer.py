from gui.visual.staticShader import StaticShader
from gui.visual.entity import Entity
import OpenGL.GL as gl
import glm

class EntityRenderer:
	def __init__(self, staticShader: StaticShader, projectionMatrix) -> None:
		self.shader = staticShader
		self.shader.start()
		self.loadProjectionMatrix(projectionMatrix)

	def render(self, entities: dict, textures: bool):
		for item in entities.items():
			self.prepareModelData(item[0], textures)
			batch = item[1]
			for entity in batch:
				self.prepareInstance(entity)
				gl.glDrawElements(gl.GL_TRIANGLES, item[0].rawModel.vertexCount, gl.GL_UNSIGNED_INT, None)
			self.unbindModelData()

	def prepareModelData(self, model, textures: bool):
		rawModel = model.rawModel
		gl.glBindVertexArray(rawModel.vaoID)
		gl.glEnableVertexAttribArray(0)
		gl.glEnableVertexAttribArray(1)
		gl.glEnableVertexAttribArray(2)

		self.shader.loadTextureOn(textures)

		if textures:
			gl.glActiveTexture(gl.GL_TEXTURE0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, model.texture.id)
		else:
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
			gl.glDisable(gl.GL_TEXTURE_2D)

	def unbindModelData(self):
		gl.glDisableVertexAttribArray(0)
		gl.glDisableVertexAttribArray(1)
		gl.glDisableVertexAttribArray(2)
		gl.glBindVertexArray(0)

	def createTransformationMatrix(self, entity: Entity):
		matrix = glm.mat4(1)
		matrix = glm.translate(matrix, entity.position)

		if entity.angle is not None:
			matrix = glm.rotate(matrix, entity.angle, entity.vector)
		elif entity.rotMatrix is not None:
			matrix = matrix * entity.rotMatrix
		else:
			matrix = glm.rotate(matrix, entity.rotX, glm.vec3(1, 0, 0))
			matrix = glm.rotate(matrix, entity.rotY, glm.vec3(0, 1, 0))
			matrix = glm.rotate(matrix, entity.rotZ, glm.vec3(0, 0, 1))

		if entity.scaleX is not None and entity.scaleY is not None and entity.scaleZ is not None:
			matrix = glm.scale(matrix, glm.vec3(entity.scaleX, entity.scaleY, entity.scaleZ))
		else:
			matrix = glm.scale(matrix, glm.vec3(entity.scale))

		return matrix

	def prepareInstance(self, entity: Entity):
		transformationMatrix = self.createTransformationMatrix(entity)
		self.shader.loadTransformationMatrix(transformationMatrix)

		if entity.color:
			self.shader.loadColor(entity.color)

	def loadProjectionMatrix(self, projectionMatrix):
		self.shader.loadProjectionMatrix(projectionMatrix)