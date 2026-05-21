import sys
#sys.path.append("..") # why was needed?
from typing import List, Dict, Any
import frams
from .creature import Creature
from gui.framsutils.framsProperty import Property
import os
import glm
from .utils import parseNeuronDToOrient

class FramsLib:
	def __init__(self):
		self.frams = frams
		self.connected = False

	def initConnection(self, frams_path: str) -> None:
		self.frams.init(frams_path, "-t") #enable ExtValue thread synchronization with "-t"
		#self.messageCatcher = self.frams.MessageCatcher.new() # experiments with catching messages so that they could be later displayed in GUI... but let's rather print them on console
		#self.messageCatcher.store = 1
		self.connected = True
		
	def closeConnection(self) -> None:
		self.connected = False

	def getError(self) -> None or str:
		#if self.messageCatcher.error_count._value() > 0:
		#	return self.messageCatcher.messages._value()
		return None

	def writeCreatureFromString(self, groupNo: int, creature: str) -> None:
		if self.connected:
			self.frams.Populations[groupNo].add(creature)

	def step(self):
		if self.connected:
			self.frams.Simulator.step()

	def start(self):
		if self.connected:
			self.frams.Simulator.start()

	def stop(self):
		if self.connected:
			self.frams.Simulator.stop()

	def info(self, path:str = "/") -> List[Property]:
		properties: List[Property] = []

		if self.connected:
			if path.startswith('/'):
				path = path[1:]

			c = 0 if path == "" else path.count("/") + 1

			if path.endswith('/'):
				path = path[:-1]

			if c > 0:
				path += '/'

			current = self.frams.sim_params
			G = current._groupCount()
			for g in range(G):
				Id = current._groupName(g)
				name = Id.rsplit(': ', 1)[-1]
				Id = Id.replace(": ", "/")
				tmpId = Id + "/"
				prop = Property(Id=Id, Name=name, Type="o")
				if(Id.count('/') == c and tmpId.startswith(path)):
					prop.p["id"] = Id.split('/')[-1]
					properties.append(prop)

		return properties

	def infoList(self, path: str) -> List[Property]:
		properties: List[Property] = []
		raise Exception("unimplemented/unwanted")
		return properties

	def _VstyleParserColor(self, style: str):
		f = "color=0x"
		idx = style.find(f)
		if idx > -1:
			hx = style[idx+len(f):idx+len(f)+6]
			r = int(hx[0:2], 16) / 255.0
			g = int(hx[2:4], 16) / 255.0
			b = int(hx[4:6], 16) / 255.0
			return r, g, b
		return None, None, None

	def readCreatures(self, groupNo: int, creatureNo: int, colors):
		ret_creatures: List[Creature] = []
		if self.connected:
			groups = [self.frams.Populations[groupNo]] if groupNo != '+' else self.frams.Populations
			for group in groups:
				creatures = [group[creatureNo]] if groupNo != '+' else group
				for c in creatures:
					creature = Creature(c.population.index._int(), c.index._int())
					numparts = c.numparts._int()
					numjoints = c.numjoints._int()
					numneurons = c.numneurons._int()

					vstyle = c.model.Vstyle._value()
					modelR, modelG, modelB = self._VstyleParserColor(vstyle)

					mechPart = [0,0,0]
					for i in range(numparts):
						if i >= c.numparts._int():
							break
						mp = c.getMechPart(i)
						mechPart = [mp.x._double(), mp.y._double(), mp.z._double()]
						creature.mechParts.append(mechPart)
						if colors:
							vstyle = mp.part.Vstyle._value()
							r, g, b = self._VstyleParserColor(vstyle)

							tr = mp.part.vr._value()
							tg = mp.part.vg._value() 
							tb = mp.part.vb._value()

							if modelR is not None:
								tr = modelR
								tg = modelG
								tb = modelB

							if r is not None:
								tr = r
								tg = g
								tb = b
							
							creature.colorsPart.append(glm.vec3(tr, tg, tb))
							
						creature.stylePart.append(mp.part.Vstyle._value())
						orient = mp.orient.toVector.toString._value()
						orient = orient[orient.find('[')+1:orient.find(']')]
						R = glm.mat4(glm.mat3(*[float(x) for x in orient.split(',')]))
						creature.partOrient.append(R)
					for i in range(numjoints):
						if i >= c.numjoints._int():
							break
						j = c.getJoint(i)
						joint = [j.p1._int(), j.p2._int()]
						creature.joints.append(joint)
						if colors:
							vstyle = mp.part.Vstyle._value()
							r, g, b = self._VstyleParserColor(vstyle)

							tr = j.vr._value()
							tg = j.vg._value() 
							tb = j.vb._value()

							if r is not None:
								tr = r
								tg = g
								tb = b
							
							creature.colorsJoint.append(glm.vec3(tr, tg, tb))

						creature.styleJoint.append(j.Vstyle._value())

					for i in range(numneurons):
						if i >= c.numneurons._int():
							break
						n = c.getNeuroDef(i)
						d = n.d._value()
						creature.neurons.append((n.p._value(), n.j._value()))
						creature.styleNeuron.append(d.split(':')[0])
						
						rorient = parseNeuronDToOrient(d)

						creature.neuronRelativeOrient.append(rorient)
					ret_creatures.append(creature)
		return ret_creatures

	def readGenePoolsGroups(self) -> Dict[str, int]:
		genotypes = {}
		if self.connected:
			for gp in self.frams.GenePools:
				genotypes[getattr(gp, "name")._value()] = getattr(gp, "index")._value()
		return genotypes

	def readGenePools(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		genotypes = {}
		if self.connected:
			for gp in self.frams.GenePools:
				group = gp.name._value()
				genotypes[group] = []
				for g in gp:
					genotype = {}
					for p in props:
						genotype[p] = getattr(g, p)._value()
					genotype["group"] = group
					genotypes[group].append(genotype)
		return genotypes

	def readPopulationsGroups(self) -> Dict[str, int]:
		creatures = {}
		if self.connected:
			for p in self.frams.Populations:
				creatures[getattr(p, "name")._value()] = getattr(p, "index")._value()

		return creatures

	def readPopulations(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		creatures = {}
		if self.connected:
			for p in self.frams.Populations:
				group = p.name._value()
				creatures[group] = []
				for c in p:
					creature = {}
					for p in props:
						creature[p] = getattr(c, p)._value()
					creature["group"] = group
					creatures[group].append(creature)
		return creatures

	def readCreatureInfo(self, groupNo: int, creatureNo: str) -> List[Property]:
		properties: List[Property] = []
		if self.connected:
			creature = next(pop for group in self.frams.Populations for pop in group if pop.uid._string() == creatureNo)
			G = creature._groupCount()
			for g in range(G):
				groupName = creature._groupName(g)
				for m in range(creature._memberCount(g)):
					p = creature._groupMember(g, m)
					t = creature._propType(p)
					if t[0] and t[0] not in "xople":
						prop = Property(creature._propId(p), creature._propName(p), t)
						prop.p["flags"] = creature._propFlags(p)
						prop.p["value"] = getattr(creature, creature._propId(p))._string()
						prop.p["help"] = creature._propHelp(p)
						prop.p["group"] = groupName
						properties.append(prop)

		return properties

	def readGenotypeInfo(self, groupNo: int, genotypeNo: str) -> List[Property]:
		properties: List[Property] = []
		if self.connected:
			genotype = next(gen for group in self.frams.GenePools for gen in group if gen.uid._string() == genotypeNo)
			G = genotype._groupCount()
			for g in range(G):
				groupName = genotype._groupName(g)
				for m in range(genotype._memberCount(g)):
					p = genotype._groupMember(g, m)
					t = genotype._propType(p)
					if t[0] and t[0] not in "xople":
						prop = Property(genotype._propId(p), genotype._propName(p), t)
						prop.p["flags"] = genotype._propFlags(p)
						prop.p["value"] = getattr(genotype, genotype._propId(p))._string()
						prop.p["help"] = genotype._propHelp(p)
						prop.p["group"] = groupName
						properties.append(prop)

		return properties

	def readPropertyInfo(self, path: str) -> List[Property]:
		properties: List[Property] = []
		if self.connected:
			if path.startswith('/'):
				path = path[1:]

			c = path.count("/")

			if path.endswith('/'):
				path = path[:-1]

			current = self.frams.sim_params
			G = current._groupCount()
			for g in range(G):
				Id = current._groupName(g).replace(": ", "/")
				groupName = current._groupName(g)
				if(Id.count('/') == c and Id == path):
					for m in range(current._memberCount(g)):
						p = current._groupMember(g, m)
						t = current._propType(p)
						if t[0] and t[0] not in "xole":
							prop = Property(current._propId(p), current._propName(p), t)
							if not prop.p["id"].startswith("_"):
								prop.p["flags"] = current._propFlags(p)
								prop.p["value"] = getattr(current, current._propId(p)) if t[0] == 'p' else getattr(current, current._propId(p))._string()
								prop.p["help"] = current._propHelp(p)
								prop.p["v"] = current
								prop.p["group"] = groupName
								properties.append(prop)
					break

		return properties

	def writeCreatureInfo(self, uid: str, prop: str, value: str) -> bool:
		if self.connected:
			creature = next(pop for group in self.frams.Populations for pop in group if pop.uid._string() == uid)
			creature.__setattr__(prop, value)
			return True
		return False

	def writeGenotypeInfo(self, uid: str, prop: str, value: str) -> bool:
		if self.connected:
			genotype = next(gen for group in self.frams.GenePools for gen in group if gen.uid._string() == uid)
			genotype.__setattr__(prop, value)
			return True
		return False

	def writePropertyInfo(self, path: str, prop: str, value: str):
		if self.connected:
			current = self.frams.sim_params
			current.__setattr__(prop, value)
			return True
		return False

	def getSimulationStatus(self) -> bool:
		if self.connected:
			return True if self.frams.Simulator.running._int() == 1 else False
		return False

	def loadFile(self, path: str) -> None:
		self.frams.Simulator.load(path)

	def importFile(self, path: str, options: int) -> None:
		self.frams.Simulator.ximport(path, options)

	def saveFile(self, path: str, options: int) -> None:
		path = os.path.relpath(path, self.frams.res_dir + "\\scripts_output")
		self.frams.Simulator.export(path, options, -1, -1)

	def getWorldType(self) -> int:
		return self.frams.World.wrldtyp._value()

	def getWorldSize(self) -> float:
		return self.frams.World.wrldsiz._value()

	def getWorldWaterLevel(self) -> float:
		return self.frams.World.wrldwat._value()

	def getWorldBoundaries(self) -> int:
		return self.frams.World.wrldbnd._value()

	def getWorldMap(self) -> str:
		return self.frams.WorldMap.getAsString("", "")._value()

	def getSimtype(self) -> int:
		return self.frams.World.simtype._value()