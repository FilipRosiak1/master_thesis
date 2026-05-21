from typing import Dict, List, Any, Tuple
from gui.framsutils.FramsSocket import FramsSocket
from gui.framsutils.creature import Creature
from gui.framsutils.framsProperty import Property
from gui.framsutils.FramsInterface import FramsInterface, TreeNode, InterfaceType

class SocketInterface(FramsInterface):
	DEBUG_ADD_FIRST_CREATURE = False
	DEBUG_FIRST_CREATURE = "X"

	def __init__(self) -> None:
		self.interfaceType = InterfaceType.SOCKET
		self.frams: FramsSocket = FramsSocket()

	def connect(self, address: str, port: int):
		self.frams.initConnection(address, port)
		if self.DEBUG_ADD_FIRST_CREATURE:
			self.frams.writeCreatureFromString(0, self.DEBUG_FIRST_CREATURE)
			self.frams.sendRequest("call /simulator/genepools/groups/0 add {}".format(self.DEBUG_FIRST_CREATURE))

	def disconnect(self):
		self.frams.closeConnection()

	def start(self):
		self.frams.start()

	def getError(self):
		return None

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
		p = parent
		path: str = " " + treeNode.node.p["id"]
		while p:
			if p.node.p["id"] != '/' and p.parent and p.parent.node.p["id"] != "/":
				path = "/" + p.node.p["id"] + path
			p = p.parent

		groupsNo = int(self.listTreeList(path)[0].p["groups"])
		path = path.replace(' ', '/') + '/'
		childs = [Property("", str(x), "o") for x in range(groupsNo)]

		for child in childs:
			child.p["id"] = child.p["name"]
			if child.p["type"][0] == "o":
				nodeChilds = self.listTree(path + child.p["id"])
				c = self._makeTree(treeNode, child, nodeChilds)
				treeNode.addChild(c)

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

	def readCreatureDetails(self, group: int, index: str):
		return self.frams.readCreatureInfo(group, index)

	def readGenotypeDetails(self, group: int, index: str):
		return self.frams.readGenotypeInfo(group, index)

	def readParameterDetails(self, prop: str):
		return self.frams.readPropertyInfo(prop)

	def writeCreatureDetail(self, creatureNo: str, prop: str, value: str):
		return self.frams.writeCreatureInfo(creatureNo, prop, value)

	def writeGenotypeDetail(self, genotypeNo: str, prop: str, value: str):
		return self.frams.writeGenotypeInfo(genotypeNo, prop, value)

	def writeParameterDetail(self, path: str, prop: str, value: str):
		return self.frams.writePropertyInfo(path, prop, value)

	def registerRunningChangeEventCallback(self, callback):
		self.frams.refreshControlButtonsCallback = callback

	def registerTreeviewRefreshEventCallback(self, callback):
		self.frams.refreshTreeviewCallback = callback

	def getMotd(self) -> str:
		try:
			tmp = self.frams.infoList("/ motd")[0].p["motd"]
		except:
			tmp = "Not connected"
		return tmp

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
		return [(0.5, 1),
				(2, 1), 
				(5, 1),
				(10, 1),
				(25, 1),
				(50, 1),
				(-1, 1)]