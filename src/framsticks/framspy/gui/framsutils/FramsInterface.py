from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import enum
from xmlrpc.client import Boolean
from gui.framsutils.creature import Creature
from gui.framsutils.framsProperty import Property
import glm

class TreeNode:
	def __init__(self, node: Property, children: List[Property] = None, parent: "TreeNode" = None) -> None:
		self.node = node
		self.parent = parent
		self.children = []
		if children is not None:
			for child in children:
				self.addChild(child)

	def addChild(self, child):
		assert isinstance(child, TreeNode)
		self.children.append(child)

	def setParent(self, parent):
		self.parent = parent

class InterfaceType(enum.Enum):
	SOCKET = 1
	LIB = 2

class FramsInterface(ABC):
	interfaceType: InterfaceType

	@abstractmethod
	def connect(self, address: str, port: int):
		raise NotImplementedError

	@abstractmethod
	def disconnect(self):
		raise NotImplementedError

	@abstractmethod
	def start(self):
		raise NotImplementedError
		
	@abstractmethod
	def getError(self):
		raise NotImplementedError
	
	@abstractmethod
	def stop(self):
		raise NotImplementedError
		
	@abstractmethod
	def step(self):
		raise NotImplementedError
		
	@abstractmethod
	def listTree(self, path="/") -> List[Property]:
		raise NotImplementedError
		
	@abstractmethod
	def listTreeList(self, path) -> List[Property]:
		raise NotImplementedError

	@abstractmethod
	def makeInfoTree(self) -> TreeNode:
		raise NotImplementedError
	
	@abstractmethod
	def readCreatures(self, colors: Boolean) -> List[Creature]:
		"""read all creatures to render"""
		raise NotImplementedError

	@abstractmethod
	def readGenePoolsGroups(self) -> Dict[str, int]:
		"""returns dict of {name, index} of gene pools groups."""
		raise NotImplementedError

	@abstractmethod
	def readGenePools(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		"""returns list of props and values for every gene pool group in format {gp group, [{prop, value}]}."""
		raise NotImplementedError

	@abstractmethod
	def readPopulationsGroups(self) -> Dict[str, int]:
		"""returns dict of {name, index} of populations groups."""
		raise NotImplementedError

	@abstractmethod
	def readPopulations(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		"""returns list of props and values for every population group in format {pop group, [{prop, value}]}."""
		raise NotImplementedError

	@abstractmethod
	def readCreatureDetails(self, group: int, index: str) -> List[Property]:
		"""read all details of creature specified by group and index."""
		raise NotImplementedError

	@abstractmethod
	def readGenotypeDetails(self, group: int, index: str) -> List[Property]:
		"""read all details of genotype specified by group and index."""
		raise NotImplementedError

	@abstractmethod
	def readParameterDetails(self, prop: str) -> List[Property]:
		"""read all property details, prop is the standard server path."""
		raise NotImplementedError

	@abstractmethod
	def writeCreatureDetail(self, creatureNo: str, prop: str, value: str):
		"""write value to property of creature."""
		raise NotImplementedError

	@abstractmethod
	def writeGenotypeDetail(self, genotypeNo: str, prop: str, value: str):
		"""write value to property of genotype."""
		raise NotImplementedError

	@abstractmethod
	def writeParameterDetail(self, path: str, prop: str, value: str):
		"""write value to property of path."""
		raise NotImplementedError

	@abstractmethod
	def getMotd(self) -> str:
		"""get message of the day."""
		raise NotImplementedError

	@abstractmethod
	def loadFile(self, path: str) -> None:
		"""load file from path into frams."""
		raise NotImplementedError

	@abstractmethod
	def importFile(self, path: str, options: int) -> None:
		"""import file from path into frams with options."""
		raise NotImplementedError

	@abstractmethod
	def saveFile(self, path: str, options: int) -> None:
		"""save file into path from frams with options."""
		raise NotImplementedError

	@abstractmethod
	def getWorldType(self) -> int:
		"""get type of the world."""
		raise NotImplementedError

	@abstractmethod
	def getWorldSize(self) -> float:
		raise NotImplementedError

	@abstractmethod
	def getWorldWaterLevel(self) -> float:
		raise NotImplementedError

	@abstractmethod
	def getWorldBoundaries(self) -> int:
		raise NotImplementedError

	@abstractmethod
	def getWorldMap(self) -> str:
		"""get geometry of the world."""
		raise NotImplementedError

	@abstractmethod
	def getSimtype(self) -> int:
		raise NotImplementedError

	@abstractmethod
	def getFPSDefinitions(self) -> List[Tuple[int, int]]:
		"""get list of the fps definitions for fps combobox
		first value determines fps limit, second number of steps per frame.
		"""
		raise NotImplementedError