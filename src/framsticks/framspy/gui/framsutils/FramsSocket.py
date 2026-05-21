from typing import List, Dict, Any, Tuple, Callable
from .comm import CommWrapper
from .creature import Creature
from gui.framsutils.framsProperty import Property
from framsfiles import reader
import threading
import asyncio
from functools import partial
import time
import glm
from .utils import parseNeuronDToOrient

class FramsSocket:
	def __init__(self):
		self.comm = CommWrapper()
		self.refreshControlButtonsCallback = None
		self.refreshTreeviewCallback = None
		self.loop = asyncio.get_event_loop()
		self.loop_thread = threading.Thread(target=self.loop.run_forever)
		self.loop_thread.start()
		
	def initConnection(self, address: str = '127.0.0.1', port: int = 9009) -> None:
		self.comm = CommWrapper()
		i = 0
		time_delta = 0.1
		total_time = 10
		self.comm.start(address, port)
		while self.comm.client.connecting == True and i < total_time / time_delta:
			time.sleep(time_delta)
			i += 1
		response = self.comm.client.connected
		if response:
			self.comm.client.consumer.runningChangeEventCallback = self._runningChangeEventCallback
			self.comm.client.consumer.populationsGroupChangeEventCallback = self._populationsGroupChangeEventCallback
			self.comm.client.consumer.genepoolsGroupChangeEventCallback = self._genepoolsGroupChangeEventCallback
			self.sendRequest("reg /simulator/populations/groups_changed")
			self.sendRequest("reg /simulator/genepools/groups_changed")
			self.sendRequest("reg /simulator/running_changed")
			self.sendRequest("reg /cli/messages")
		else:
			self.comm.close()
			raise ConnectionError()

	def closeConnection(self) -> None:
		if self.comm:
			self.comm.stop()
			self.loop.call_soon_threadsafe(self.loop.stop)
			self.loop_thread.join()

	def sendRequest(self, request: str, timeout = 1.0, return_index = False, index=-1) -> List[str]:
		response = []

		if self.comm.connected:
			if self.comm:
				idx = self.comm.write(request, index=index)
				future = asyncio.run_coroutine_threadsafe(self.comm.read(idx, timeout), self.loop)
				try:
					response = future.result(timeout)
					response = [e+'\n' for e in response.split('\n')]
				except asyncio.TimeoutError:
					future.cancel()
				except Exception as ex:
					print("dropped request id {}, exception: {}".format(idx, ex))

		append = False
		change = False
		for i in range(len(response) - 2, -1, -1):
			if response[i].find('~\n') >= 0 and response[i].find('\~\n') < 0:
				response[i].replace('~\n', '')
				change = True

			if append and not change:
				response[i] = response[i] + response.pop(i + 1) #merge middle line
			elif append and change:
				response[i] = response[i] + response.pop(i + 1) #merge last line

			if change:
				append = not append
				change = False
		if return_index:
			return response, idx
		return response

	def writeCreatureFromString(self, groupNo: int, creature: str) -> None:
		creatureString = "call /simulator/populations/groups/{} createFromString {}".format(groupNo, creature)
		self.sendRequest(creatureString)

	def step(self):
		self.sendRequest("call /simulator step")

	def start(self):
		self.sendRequest("call /simulator start")

	def stop(self):
		self.sendRequest("call /simulator stop")

	def info(self, path: str = "/") -> List[Property]:
		response = self.sendRequest("info {}".format(path))
		properties = self._infoParser(response)
		return properties

	def infoList(self, path: str) -> List[Property]:
		response = self.sendRequest("get {}".format(path))
		properties = self._infoParser(response)
		return properties

	def _infoParser(self, response: List[str], packToProperty = True) -> List[Property]:
		properties: List[Property] = []

		tmp = reader.loads(''.join(response))
		if packToProperty:
			properties = [Property(p=x) for x in tmp]
			return properties
		return tmp

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

	def readCreatures(self, groupNo: int, creatureNo: int, color):
		creatures: List[Creature] = []
		if color:
			creatureString = "get /simulator/populations/groups/{}/creatures/{}(/mechparts{{x,y,z,orient}},/joints{{p1,p2,vr,vg,vb,Vstyle}},/parts{{vr,vg,vb,Vstyle}},/neurodefs{{p,j,d}},{{model_Vstyle}})".format(groupNo, creatureNo)
		else:
			creatureString = "get /simulator/populations/groups/{}/creatures/{}(/mechparts{{x,y,z,orient}},/joints{{p1,p2}},/neurodefs{{p,j,d}})".format(groupNo, creatureNo)
		response = self.sendRequest(creatureString)

		mechParts = []
		joints = []
		partColors = []
		neurons = []
		neuronStyles = []
		creature: Creature = None

		mechPart: List[float] = [0,0,0]
		joint: List[int] = [0,0]
		partColor: glm.vec3 = glm.vec3(0,0,0)
		neuron: Tuple(int, int) = (-1,-1)

		files = [i for i, x in enumerate(response) if x.startswith("file")]
		files.append(-1)

		for i in range(len(files)-1):
			file = response[files[i]]

			groupsStr = "groups/"
			groupBegin = file.find(groupsStr) + len(groupsStr)
			groupEnd = file.find("/", groupBegin)
			group = int(file[groupBegin:groupEnd])

			creaturesStr = "creatures/"
			creatureIndexBegin = file.find(creaturesStr) + len(creaturesStr)
			creatureIndexEnd = file.find("/", creatureIndexBegin)
			creatureIndex = int(file[creatureIndexBegin:creatureIndexEnd])

			creature = next((x for x in creatures if x.group == group and x.index == creatureIndex), None)
			if not creature:
				creature = Creature(group, creatureIndex)
				creatures.append(creature)

			mechParts = []
			joints = []
			partColors = []
			jointColors = []
			neurons = []
			neuronStyles = []
			partOrient = []
			neuroRelativeOrient = []

			resp = self._infoParser(response[files[i]:files[i+1]], False)

			for p in resp:
				if p["_classname"] == "MechPart":
					mechPart = [p['x'], p['y'], p['z']]
					orient = p["orient"]
					orient = orient[orient.find('[')+1:orient.find(']')]
					orient = glm.mat4(glm.mat3(*[float(x) for x in orient.split(',')]))
					mechParts.append(mechPart)
					partOrient.append(orient)
				elif p["_classname"] == "Joint":
					joint = [p["p1"], p["p2"]]
					joints.append(joint)
					
					if color:
						jointColor = glm.vec3(p["vr"], p["vg"], p["vb"])
						colors = self._VstyleParserColor(p["Vstyle"])
						if colors and colors[0] != None:
							jointColor = glm.vec3(colors)

						jointColors.append(jointColor)
				elif p["_classname"] == "Part":
					partColor = glm.vec3(p["vr"], p["vg"], p["vb"])

					colors = self._VstyleParserColor(p["Vstyle"])
					if colors and colors[0] != None:
						partColor = glm.vec3(colors)

					partColors.append(partColor)
				elif p["_classname"] == "NeuroDef":
					neuron = (p['p'], p['j'])
					neurons.append(neuron)
					neuronStyles.append(p['d'].split(':')[0])

					'''ix = p['d'].find(':')
					rorient = glm.mat4(1)
					if ix != -1:
						ds = p['d'][ix+1:].split(',')
						ds = [x.split('=') for x in ds]
						rots = [x for x in ds if x[0].startswith('r')]

						if not any("ry" in x[0] for x in rots):
							rots.append(["ry", '0'])
						if not any("rz" in x[0] for x in rots):
							rots.append(["rz", '0'])

						angle = next(float(x[1]) for x in rots if x[0] == "rz")
						rorient = glm.rotate(rorient, angle, glm.vec3(0,0,1))
						angle = -next(float(x[1]) for x in rots if x[0] == "ry")
						rorient = glm.rotate(rorient, angle, glm.vec3(0,1,0))'''
						
					rorient = parseNeuronDToOrient(p['d'])
					neuroRelativeOrient.append(rorient)
				elif p["_classname"] == "Creature":
					colors = self._VstyleParserColor(p["model_Vstyle"])
					if colors and colors[0] != None:
						creature.colorsPart = [glm.vec3(colors) for i in creature.colorsPart]

			creature.mechParts.extend(mechParts.copy())
			creature.joints.extend(joints.copy())
			creature.colorsPart.extend(partColors.copy())
			creature.colorsJoint.extend(jointColors.copy())
			creature.neurons.extend(neurons.copy())
			creature.styleNeuron.extend(neuronStyles.copy())
			creature.partOrient.extend(partOrient.copy())
			creature.neuronRelativeOrient.extend(neuroRelativeOrient.copy())

		invalidCreatures = []
		for creature in creatures:
			if len(creature.mechParts) == 0:
				invalidCreatures.append(creature)
		for creature in invalidCreatures:
			creatures.remove(creature)

		return creatures

	def readGenePoolsGroups(self) -> Dict[str, int]:
		genotypes: Dict[str, int] = {}
		tmp = []
		for i in range(3):
			response = self.sendRequest("get /simulator/genepools/groups/+ index,name")
			tmp = self._infoParser(response)
			if len(tmp) > 0:
				break

		for i in tmp:
			genotypes[i.p["name"]] = i.p["index"]

		return genotypes

	def readGenePools(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		genotypes: Dict[str, List[Dict[str, Any]]] = {}

		for i in range(3):
			response = self.sendRequest("get /simulator/genepools/groups/+ index,name")
			tmp = self._infoParser(response)
			if len(tmp) > 0:
				break
		else:
			return genotypes

		tu = {}
		for i in tmp:
			tu[i.p["index"]] = i.p["name"]
			genotypes[i.p["index"]] = []

		p = ",".join(props)
		response = self.sendRequest("get /simulator/genepools/groups/+/genotypes/+ {}".format(p))

		files = [i for i, x in enumerate(response) if x.startswith("file")]
		files.append(-1)

		for i in range(len(files)-1):
			file = response[files[i]]

			groupsStr = "groups/"
			groupBegin = file.find(groupsStr) + len(groupsStr)
			groupEnd = file.find("/", groupBegin)
			group = int(file[groupBegin:groupEnd])

			genotypesStr = "genotypes/"
			genotypeIndexBegin = file.find(genotypesStr) + len(genotypesStr)
			genotypeIndexEnd = file.find("/", genotypeIndexBegin-1) +2
			genotypeIndex = int(file[genotypeIndexBegin:genotypeIndexEnd])

			resp = self._infoParser(response[files[i]:files[i+1]], False)

			genotypes[group].append(resp[0])
			genotypes[group][-1].update({"group": group, "index": genotypeIndex})

		g = {}
		for (i, gen) in genotypes.items():
			g[tu[i]] = gen

		return g

	def readPopulationsGroups(self) -> Dict[str, int]:
		creatures: Dict[str, int] = {}
		tmp = []
		for i in range(3):
			response = self.sendRequest("get /simulator/populations/groups/+ index,name")
			tmp = self._infoParser(response)
			if len(tmp) > 0:
				break

		for i in tmp:
			creatures[i.p["name"]] = i.p["index"]

		return creatures

	def readPopulations(self, props: List[str]) -> Dict[str, List[Dict[str, Any]]]:
		creatures: Dict[str, List[Dict[str, Any]]] = {}
		for i in range(3):
			response = self.sendRequest("get /simulator/populations/groups/+ index,name")
			tmp = self._infoParser(response)
			if len(tmp) > 0:
				break
		else:
			return creatures

		tu = {}
		for i in tmp:
			tu[i.p["index"]] = i.p["name"]
			creatures[i.p["index"]] = []

		p = ",".join(props)
		response = self.sendRequest("get /simulator/populations/groups/+/creatures/+ {}".format(p))

		files = [i for i, x in enumerate(response) if x.startswith("file")]
		files.append(-1)

		for i in range(len(files)-1):
			file = response[files[i]]

			groupsStr = "groups/"
			groupBegin = file.find(groupsStr) + len(groupsStr)
			groupEnd = file.find("/", groupBegin)
			group = int(file[groupBegin:groupEnd])

			creaturesStr = "creatures/"
			creatureIndexBegin = file.find(creaturesStr) + len(creaturesStr)
			creatureIndexEnd = file.find("/", creatureIndexBegin-1) +2
			creatureIndex = int(file[creatureIndexBegin:creatureIndexEnd])

			resp = self._infoParser(response[files[i]:files[i+1]], False)

			creatures[group].append(resp[0])
			creatures[group][-1].update({"group": group, "index": creatureIndex})

		c = {}
		for (i, cre) in creatures.items():
			c[tu[i]] = cre

		return c

	def readCreatureInfo(self, groupNo: int, creatureNo: int) -> List[Property]:
		response = self.sendRequest("info /simulator/populations/groups/{}/creatures/{}".format(groupNo, creatureNo))
		response2 = self.sendRequest("get /simulator/populations/groups/{}/creatures/{}".format(groupNo, creatureNo))

		tmpProperties = self._infoParser(response)
		tmpValues = self._infoParser(response2)
		properties: List[Property] = []
		for prop in tmpProperties:
			if "type" in prop.p and "id" in prop.p and len(prop.p["type"]) > 0:
				if prop.p["type"][0] not in "xole": #xol is a list of unwanted types
					if prop.p["type"][0] == 'p':
						prop.p["value"] = partial(self.call, "/simulator/populations/groups/{}/creatures/{} {}".format(groupNo, creatureNo, prop.p["id"]))
					else:
						prop.p["value"] = next(v for k, v in tmpValues[0].p.items() if k == prop.p["id"])
					properties.append(prop)

		return properties

	def readGenotypeInfo(self, groupNo: int, genotypeNo: int) -> List[Property]:
		response = self.sendRequest("info /simulator/genepools/groups/{}/genotypes/{}".format(groupNo, genotypeNo))
		response2 = self.sendRequest("get /simulator/genepools/groups/{}/genotypes/{}".format(groupNo, genotypeNo))

		tmpProperties = self._infoParser(response)
		tmpValues = self._infoParser(response2)
		properties: List[Property] = []
		for prop in tmpProperties:
			if "type" in prop.p and "id" in prop.p and len(prop.p["type"]) > 0:
				if prop.p["type"][0] not in "xole": #xol is a list of unwanted types
					if prop.p["type"][0] == 'p':
						prop.p["value"] = partial(self.call, "/simulator/genepools/groups/{}/genotypes/{} {}".format(groupNo, genotypeNo, prop.p["id"]))
					else:
						prop.p["value"] = next(v for k, v in tmpValues[0].p.items() if k == prop.p["id"])
					properties.append(prop)

		return properties

	def readPropertyInfo(self, path) -> List[Property]:
		response = self.sendRequest("info {}".format(path))
		response2 = self.sendRequest("get {}".format(path))

		tmpProperties = self._infoParser(response)
		tmpValues = self._infoParser(response2)
		properties: List[Property] = []
		for prop in tmpProperties:
			if "type" in prop.p and "id" in prop.p and len(prop.p["type"]) > 0:
				if prop.p["type"][0] and prop.p["type"][0] not in "xole": #xole is a list of unwanted types
					if prop.p["type"][0] == 'p':
						prop.p["value"] = partial(self.call, "{} {}".format(path, prop.p["id"]))
					else:
						prop.p["value"] = next((v for k, v in tmpValues[0].p.items() if k == prop.p["id"]), "")
					properties.append(prop)

		return properties

	def writeCreatureInfo(self, uid: str, prop: str, value: str) -> bool:
		response = self.sendRequest("get /simulator/populations size")
		size = next(int(x[x.find(":")+1:-1]) for x in response if x.startswith("size"))
		if response:
			for i in range(size):
				res = self.sendRequest("set /simulator/populations/groups/{}/creatures/{} {} \"{}\"".format(i, uid, prop, value))
			return True
		return False

	def writeGenotypeInfo(self, uid: str, prop: str, value: str) -> bool:
		response = self.sendRequest("get /simulator/genepools size")
		size = next(int(x[x.find(":")+1:-1]) for x in response if x.startswith("size"))
		if response:
			for i in range(size):
				res = self.sendRequest("set /simulator/genepools/groups/{}/genotypes/{} {} \"{}\"".format(i, uid, prop, value))
			return True
		return False

	def writePropertyInfo(self, path: str, prop: str, value: str):
		response = self.sendRequest("set {} {} \"{}\"".format(path, prop, value))
		if response:
			return True
		return False

	def call(self, path: str):
		self.sendRequest("call {}".format(path))

	def _runningChangeEventCallback(self, block: str, header: str):
		prop = self._infoParser([block])[0]
		if self.refreshControlButtonsCallback:
			self.refreshControlButtonsCallback(prop.p["value"])
		
	def _populationsGroupChangeEventCallback(self, block: str, header: str):
		if self.refreshTreeviewCallback:
			self.refreshTreeviewCallback()

	def _genepoolsGroupChangeEventCallback(self, block: str, header: str):
		if self.refreshTreeviewCallback:
			self.refreshTreeviewCallback()

	def registerMessageEventCallback(self, callback: Callable[[str, str], None]):
		self.comm.client.consumer.messagesEventCallback = callback

	def loadFile(self, path: str) -> None:
		response, idx = self.sendRequest("call /simulator netload", return_index=True)
		if response:
			if response[0].startswith("needfile"):
				with open(path) as file:
					data = "file \n" + file.read() + "eof"
					self.sendRequest(data, index=idx)

	def importFile(self, path: str, options: int) -> None:
		response, idx = self.sendRequest("call /simulator netimport {}".format(options), return_index=True)
		if response:
			if response[0].startswith("needfile"):
				with open(path) as file:
					data = "file \n" + file.read() + "eof"
					self.sendRequest(data, index=idx)

	def saveFile(self, path: str, options: int) -> None:
		response = self.sendRequest("call /simulator netexport {} -1 -1".format(options))
		if response:
			start = 0
			end = len(response) - 1
			if response[0].startswith("file"):
				start = 1

			while response[end].strip() == '':
				end -= 1

			if response[end].startswith("ok"):
				end -= 1
			if response[end].startswith("eof"):
				end -= 1

			while response[end].strip() == '':
				end -= 1
			end += 1

		with open(path, 'w') as file:
			file.write(''.join(response[start:end]))
			file.write("\n")

	def getWorldType(self) -> int:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world wrldtyp")
			info = self._infoParser(response)
		return info[0].p["wrldtyp"]

	def getWorldSize(self) -> float:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world wrldsiz")
			info = self._infoParser(response)
		return info[0].p["wrldsiz"]

	def getWorldWaterLevel(self) -> float:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world wrldwat")
			info = self._infoParser(response)
		return info[0].p["wrldwat"]

	def getWorldBoundaries(self) -> int:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world wrldbnd")
			info = self._infoParser(response)
		return info[0].p["wrldbnd"]

	def getWorldMap(self) -> str:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world geometry")
			info = self._infoParser(response)
		return info[0].p["geometry"]

	def getSimtype(self) -> int:
		info = []
		while not info:
			response = self.sendRequest("get /simulator/world simtype")
			info = self._infoParser(response)
		return info[0].p["simtype"]