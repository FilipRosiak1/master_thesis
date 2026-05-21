from typing import Dict, List, Any, Tuple
from gui.framsutils.FramsLib import FramsLib
from gui.framsutils.creature import Creature
from gui.framsutils.framsProperty import Property
from gui.framsutils.FramsInterface import FramsInterface, TreeNode, InterfaceType

class LibInterface(FramsInterface):
	DEBUG_ADD_FIRST_CREATURE = False
	DEBUG_FIRST_CREATURE = "fffX(fffIX[S:-1.556],RRlFFFMMMX[|-1:-10,3:10]llllFFMMMX[|1:-4.450]RlllFFFMMMX[|G:-1.404],fffIX[S:1])"

	def __init__(self) -> None:
		self.interfaceType = InterfaceType.LIB
		self.frams: FramsLib = FramsLib()

	def connect(self, address: str, port: int):
		self.frams.initConnection(address)
		if self.DEBUG_ADD_FIRST_CREATURE:
			self.frams.writeCreatureFromString(0, self.DEBUG_FIRST_CREATURE)
			self.frams.frams.GenePools[0].add(self.DEBUG_FIRST_CREATURE)

	def disconnect(self):
		self.frams.closeConnection()

	def getError(self) -> None or str:
		return self.frams.getError()

	def start(self):
		self.frams.start()

	def stop(self):
		self.frams.stop()

	def step(self):
		self.frams.step()

	def listTree(self, path="/") -> List[Property]:
		return self.frams.info(path)

	def listTreeList(self, path) -> List[Property]:
		return self.frams.infoList(path)

	def makeInfoTree(self) -> TreeNode:
		root = Property("", "/", "o")
		childs = self.listTree()
		tree = self._makeTree(None, root, childs)
		tree.node.p["id"] = "/"
		return tree
	
	def _makeTree(self, parent: TreeNode, node: Property, childs: List[Property]) -> TreeNode:
		treeNode = TreeNode(node, None, parent)
		p = parent
		path = treeNode.node.p["id"]
		while p:
			if p.node.p["id"] != '/':
				path = p.node.p["id"] + "/" + path
			p = p.parent

		for child in childs:
			if child.p["id"] and child.p["name"] and child.p["type"]:
				if child.p["type"][0] == "o":
					nodeChilds = self.listTree(path + "/" + child.p["id"])
					c = self._makeTree(treeNode, child, nodeChilds)
					treeNode.addChild(c)
				elif child.p["type"][0] == "l" and child.p["id"] == "groups":
					nodeChilds = self.listTree(path + "/" + child.p["id"])
					c = self._makeTreeFromList(treeNode, child, nodeChilds)
					treeNode.addChild(c)
		return treeNode

	def _makeTreeFromList(self, parent: TreeNode, node: Property, childs: List[Property]) -> TreeNode:
		treeNode = TreeNode(node, None, parent)
		return treeNode

	def readCreatures(self, colors = True) -> List[Creature]:
		return self.frams.readCreatures("+", "+", colors)

	def readGenePoolsGroups(self) -> Dict[str, int]:
		return self.frams.readGenePoolsGroups()

	def readGenePools(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		return self.frams.readGenePools(props)

	def readPopulationsGroups(self) -> Dict[str, int]:
		return self.frams.readPopulationsGroups()

	def readPopulations(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		return self.frams.readPopulations(props)

	def readCreatureDetails(self, group: int, index: int):
		return self.frams.readCreatureInfo(group, index)

	def readGenotypeDetails(self, group: int, index: int):
		return self.frams.readGenotypeInfo(group, index)

	def readParameterDetails(self, prop: str):
		return self.frams.readPropertyInfo(prop)

	def writeCreatureDetail(self, creatureNo: str, prop: str, value: str):
		return self.frams.writeCreatureInfo(creatureNo, prop, value)

	def writeGenotypeDetail(self, genotypeNo: str, prop: str, value: str):
		return self.frams.writeGenotypeInfo(genotypeNo, prop, value)

	def writeParameterDetail(self, path: str, prop: str, value: str):
		return self.frams.writePropertyInfo(path, prop, value)

	def getMotd(self) -> str:
		return ""

	def getSimulationStatus(self) -> bool:
		return self.frams.getSimulationStatus()

	def loadFile(self, path: str) -> None:
		self.frams.loadFile(path)

	def importFile(self, path: str, options: int) -> None:
		self.frams.importFile(path, options)

	def saveFile(self, path: str, options: int) -> None:
		self.frams.saveFile(path, options)

	def getWorldType(self) -> int:
		return self.frams.getWorldType()

	def getWorldSize(self) -> float:
		return self.frams.getWorldSize()

	def getWorldWaterLevel(self) -> float:
		return self.frams.getWorldWaterLevel()

	def getWorldBoundaries(self) -> int:
		return self.frams.getWorldBoundaries()

	def getWorldMap(self) -> str:
		return self.frams.getWorldMap()

	def getSimtype(self) -> int:
		return self.frams.getSimtype()

	def getFPSDefinitions(self) -> List[Tuple[int, int]]:
		return [(2, 1), 
				(5, 1),
				(10, 1),
				(25, 1),
				(50, 1),
				(-1, 1),
				(-1, 2),
				(-1, 3),
				(-1, 4),
				(-1, 10),
				(-1, 100),
				(-1, 1000)]