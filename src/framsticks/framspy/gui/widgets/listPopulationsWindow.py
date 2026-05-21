import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict
from gui.widgets.propertyWindow import PropertyWindow
from gui.framsutils.FramsInterface import FramsInterface
from functools import partial
import operator
from threading import Semaphore

class ListPopulationsWindow(tk.Toplevel):
	TREEVIEW_WIDTH = 600

	class FrameData:
		def __init__(self, frame, treeview, prev, index, text) -> None:
			self.frame = frame
			self.treeview = treeview
			self.prev = prev
			self.index = index
			self.text = text

	fields = ["name", "genotype", "energy", "lifespan", "uid"]
	headers = ["Name", "Genotype", "Energy", "Lifespan"]
	headers_width = [0.3, 0.5, 0.1, 0.1]

	def __init__(self, parent, posX, posY, height, refreshRate: int, framsSemaphore: Semaphore, frams: FramsInterface):
		super().__init__(parent)
		self.parent = parent
		#self.transient(parent)
		self.protocol("WM_DELETE_WINDOW", self._dismiss)
		self.title("Populations")

		self.frams: FramsInterface = None
		self.semaphore: Semaphore = framsSemaphore
		self.frams = frams

		self.update()
		self.height = height - self.winfo_rooty() + self.winfo_y()
		self.posX = posX

		self.refreshRate = refreshRate

		#MAIN SECTION
		frame_main = tk.Frame(master=self)
		self.notebook_notebook = ttk.Notebook(frame_main)
		self.frames: Dict[str, self.FrameData] = {}

		#WINDOW
		self.notebook_notebook.grid(row=0, column=0, sticky="NSEW")
		frame_main.columnconfigure(0, weight=1, minsize=0)
		frame_main.rowconfigure(0, weight=1, minsize=0)
		frame_main.grid(row=0, column=0, sticky="NSEW")

		self.columnconfigure(0, weight=1, minsize=self.TREEVIEW_WIDTH)
		self.rowconfigure(0, weight=1, minsize=self.height)

		self.update()
		width = self.winfo_width()
		self.geometry("%dx%d+%d+%d" % (width, self.height, posX, posY))
		#self.update()
		self.refresh = False
		self.dismissed = False
		self.opened = True

	def clearList(self):
		self.update()

	def refreshPopulations(self):
		if not self.dismissed and self.frams:
			self.semaphore.acquire()
			populationsDict = self.frams.readPopulations(self.fields)
			groupIndexes = self.frams.readPopulationsGroups()
			self.semaphore.release()

			groups = set(populationsDict.keys())
			groups_prev = set(self.frames.keys())
			groups_remove = groups_prev - groups
			for g in groups_remove:
				f = self.frames.pop(g)
				self.notebook_notebook.forget(f.frame)
			groups_add = groups - groups_prev
			for k, v in (sorted(groupIndexes.items(), key=lambda item: item[1])):
				if k in groups_add:
					self._addGroup(k, v)

			for (group, populations) in populationsDict.items():
				if populations or (not populations and not self.frames[group].prev):
					prevPop = set(self.frames[group].treeview.get_children())
					popUids = {x["uid"] for x in populations}

					popToRemove = prevPop - popUids
					popToAdd = popUids - prevPop
					popToUpdate = popUids.intersection(prevPop)

					#first remove unwanted items to reduce time of further operations
					for pop in popToRemove:
						self.frames[group].treeview.delete(pop)
					
					#update remaining items
					for pop in popToUpdate:
						rec = next(x for x in populations if x["uid"] == pop)
						self.frames[group].treeview.item(pop, text="{}-{}".format(rec["group"], rec["uid"]), values=self._makeValue(rec))
					
					#add new items 
					for pop in popToAdd:
						rec = next(x for x in populations if x["uid"] == pop)
						self.frames[group].treeview.insert(parent="", index="end", iid=rec["uid"], text="{}-{}".format(rec["group"], rec["uid"]), values=self._makeValue(rec))
				self.frames[group].prev = len(populations)

			if self.refresh:
				self.after(self.refreshRate + 1, self.refreshPopulations)

	def _makeValue(self, rec):
		ret = []
		for i, label in enumerate(self.fields):
			if isinstance(rec[label], str):
				ret.append(rec[label].replace('\n', "\\n"))
			else:
				ret.append(rec[label])
			if i == len(self.headers) - 1:
				break

		return tuple(ret)

	def _treeviewSelect(self, event):
		itemId = event.widget.focus()
		if itemId and event.widget.identify_row(event.y) == itemId:
			item = event.widget.item(itemId)
			if item:
				group, uid = item["text"].split('-')
				info = partial(self.frams.readCreatureDetails, group, uid)
				update = partial(self.frams.writeCreatureDetail, uid)
				cw = PropertyWindow(self, "Creature data", self.posX, info, update, self.frams.getError, self.frams, self.semaphore)

	def _dismiss(self):
		self.dismissed = True
		self.grab_release()
		self.destroy()
		self.opened = False

	def _addGroup(self, g, index):
		masterFrame = tk.Frame(master=self)
		masterFrame.columnconfigure(0, weight=1)
		masterFrame.columnconfigure(1, weight=0)
		masterFrame.rowconfigure(0, weight=1)
		
		treeview = ttk.Treeview(master=masterFrame, columns=tuple(self.headers), selectmode="browse")
		scrollbar_treeview = ttk.Scrollbar(master=masterFrame, orient=tk.VERTICAL, command=treeview.yview)
		treeview.configure(yscrollcommand=scrollbar_treeview.set)
		treeview['show'] = 'headings'
		for h, w in zip(self.headers, self.headers_width):
			treeview.heading(h, text=h)
			treeview.column(h, stretch=tk.YES, width=int(w*self.TREEVIEW_WIDTH))
		treeview.bind("<Double-1>", self._treeviewSelect) #on double left click callback 

		treeview.grid(row=0, column=0, sticky="NSEW")
		scrollbar_treeview.grid(row=0, column=1, sticky="NSEW")
		masterFrame.grid(row=0, column=0, sticky="NSEW")

		self.frames[g] = self.FrameData(masterFrame, treeview, 0, index, g)

		selected = self.notebook_notebook.select()

		self.notebook_notebook.add(self.frames[g].frame, text=g)

		#hide all tabs
		for f in self.frames.values():
			self.notebook_notebook.forget(f.frame)

		#show all tabs sorted
		for frame in (sorted(self.frames.values(), key=operator.attrgetter("index"))):
			self.notebook_notebook.add(frame.frame, text=frame.text)

		if selected:
			self.notebook_notebook.select(selected)
		else:
			self.notebook_notebook.select(self.frames[g].frame)