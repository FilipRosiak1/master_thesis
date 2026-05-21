import tkinter as tk
import tkinter.ttk as ttk
from tkinter import simpledialog
from typing import Tuple
from framsfiles import reader
from collections import Counter

class ImportWindow(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "filename" in kw:
            self.__filename = kw["filename"]
            del kw["filename"]
        else:
            self.__filename = None

        self.vec = [tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar()]

        simpledialog._QueryDialog.__init__(self, "Select import options for '{}'".format(self.__filename.split("/")[-1]), "options", *args, **kw)

    def body(self, master):
        objectCounters = self.__countOccurrences(self.__filename)

        text = "Import genotypes - {} genotypes found".format(objectCounters[0])
        cb = ttk.Checkbutton(master, text=text, variable=self.vec[0], state="enabled" if objectCounters[0] > 0 else "disabled")
        cb.pack(anchor=tk.W)

        text = "Import simulator parameters without changing expdef - {} parameter sets found".format(objectCounters[1])
        cb = ttk.Checkbutton(master, text=text, variable=self.vec[1], state="enabled" if objectCounters[1] > 0 else "disabled")
        cb.pack(anchor=tk.W)

        text = "Import genepool settings - {} gene pools found".format(objectCounters[2])
        cb = ttk.Checkbutton(master, text=text, variable=self.vec[2], state="enabled" if objectCounters[2] > 0 else "disabled")
        cb.pack(anchor=tk.W)

        text = "Import population settings - {} populations found".format(objectCounters[3])
        cb = ttk.Checkbutton(master, text=text, variable=self.vec[3], state="enabled" if objectCounters[3] > 0 else "disabled")
        cb.pack(anchor=tk.W)

        cb = ttk.Checkbutton(master, text="Create new groups for imported genepools and populations", variable=self.vec[4])
        tk.Label(master, text="").pack(anchor=tk.W)

        cb.pack(anchor=tk.W)

        cb = ttk.Checkbutton(master, text="Allow switching to a different expdef while importing simulator parameters", variable=self.vec[5])
        cb.pack(anchor=tk.W)
        
        return cb

    def getresult(self):
        options = 0
        for i, v in enumerate(self.vec):
            options += v.get() * (2 ** (i + 1))
        return options

    def __countOccurrences(self, filename) -> Tuple[int, int, int, int]:
        objects = ["org", "sim_params", "GenePool", "Population"]

        file = reader.load(filename)
        counts = Counter(p["_classname"] for p in file)
        result = [counts[o] for o in objects]

        return tuple(result)