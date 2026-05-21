import glm
from typing import List, Tuple

class Creature:
	def __init__(self, group: int, index: int) -> None:
		self.mechParts = []
		self.joints = []
		self.neurons: List[Tuple(int, int)] = []
		self.colorsJoint: List[glm.vec3] = []
		self.colorsPart: List[glm.vec3] = []
		self.stylePart: List[str] = []
		self.styleJoint: List[str] = []
		self.styleNeuron: List[str] = []
		self.partOrient: List[glm.mat4] = []
		self.neuronRelativeOrient: List[glm.mat3] = []
		self.group: int = group
		self.index: int = index

	def jointTranslation(self, index: int):
		p1 = glm.vec3(self.mechParts[self.joints[index][0]])
		return p1

	def jointLength(self, index: int):
		p1 = glm.vec3(self.mechParts[self.joints[index][0]])
		p2 = glm.vec3(self.mechParts[self.joints[index][1]])
		return glm.length(p2 - p1)

	def jointRotation(self, index: int) -> glm.mat4:
		p1 = glm.vec3(self.mechParts[self.joints[index][0]])
		p2 = glm.vec3(self.mechParts[self.joints[index][1]])

		z_axis = None
		along_d = p2 - p1
		along_o = self._lookAt(along_d)
		along_z = self._revTransform(*along_o, self.partOrient[self.joints[index][1]][2])

		if along_z.y * along_z.y + along_z.z * along_z.z > 0.5*0.5:
			z_axis = glm.vec3(self.partOrient[self.joints[index][1]][2])
		else:
			z_axis = glm.vec3(self.partOrient[self.joints[index][1]][1])
		orient = self._lookAt2(along_d, z_axis)
		return glm.mat4(glm.mat3(*orient))

	def partTranslation(self, index: int):
		return glm.vec3(self.mechParts[index])

	def partRotationMatrix(self, index: int):
		return self.partOrient[index]

	def _lookAt(self, X: glm.vec3) -> Tuple[glm.vec3, glm.vec3, glm.vec3]:
		x = glm.normalize(X)

		ax = abs(x.x)
		ay = abs(x.y)
		az = abs(x.z)
		if (ax <= ay) and (ax <= az):
			y = glm.vec3(0, -x.z, x.y)
		elif (ay <= ax) and (ay <= az):
			y = glm.vec3(-x.z, 0, x.x)
		else:
			y = glm.vec3(-x.y, x.x, 0)
		y = glm.normalize(y)
		z = glm.cross(x, y)
		return x, y, z

	def _lookAt2(self, X: glm.vec3, d: glm.vec3) -> Tuple[glm.vec3, glm.vec3, glm.vec3]:
		x = glm.normalize(X)
		y = glm.cross(d, x)
		z = glm.cross(x, y)
		if glm.length(y) < 1e-50 or glm.length(z) < 1e-50:
			return self._lookAt(X)
		
		y = glm.normalize(y)
		z = glm.normalize(z)
		return x, y, z

	def _revTransform(self, x: glm.vec3, y: glm.vec3, z: glm.vec3, s: glm.vec3) -> glm.vec3:
		return glm.vec3(
			s.x * x.x + s.y * x.y + s.z * x.z,
			s.x * y.x + s.y * y.y + s.z * y.z,
			s.x * z.x + s.y * z.y + s.z * z.z
		)