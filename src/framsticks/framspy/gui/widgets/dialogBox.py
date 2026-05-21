import tkinter as tk
import tkinter.ttk as ttk
from tkinter import simpledialog, filedialog

class SliderDialogBox(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "show" in kw:
            self.__show = kw["show"]
            del kw["show"]
        else:
            self.__show = None
        if "min" in kw:
            self.__min = kw["min"]
            del kw["min"]
        else:
            self.__min = 0
        if "max" in kw:
            self.__max = kw["max"]
            del kw["max"]
        else:
            self.__max = 100
        if "interval" in kw:
            self.__interval = kw["interval"]
            del kw["interval"]
        else:
            self.__interval = 1
        if "init_val" in kw:
            self.__init_val = kw["init_val"]
            del kw["init_val"]
        else:
            self.__init_val = None
        if "hint" in kw:
            self.hint = kw["hint"]
            del kw["hint"]
        else:
            self.hint = None

        simpledialog._QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        self.entry = tk.Scale(master=master, from_=self.__min, to=self.__max, orient=tk.HORIZONTAL, resolution=self.__interval)
        if self.__init_val:
            self.entry.set(self.__init_val)
        if self.hint:
            hint = tk.Label(master=master, text=self.hint)
            hint.grid(row=0, padx=5, sticky="WE")
        self.entry.grid(row=1, padx=5, sticky="WE")
        if self.__show is not None:
            self.entry.configure(show=self.__show)
        return self.entry

    def getresult(self):
        return self.entry.get()

class ComboboxDialogBox(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "show" in kw:
            self.__show = kw["show"]
            del kw["show"]
        else:
            self.__show = None
        if "values" in kw:
            self.__values = kw["values"]
            del kw["values"]
        else:
            self.__ = None
        if "init_val" in kw:
            self.__init_val = kw["init_val"]
            del kw["init_val"]
        else:
            self.__init_val = None
        if "hint" in kw:
            self.hint = kw["hint"]
            del kw["hint"]
        else:
            self.hint = None

        simpledialog._QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        self.entry = ttk.Combobox(master=master, values=self.__values, state="readonly")
        if self.__init_val:
            self.entry.set(self.__init_val)
        if self.hint:
            hint = tk.Label(master=master, text=self.hint)
            hint.grid(row=0, padx=5, sticky="WE")
        self.entry.grid(row=1, padx=5, sticky="WE")
        if self.__show is not None:
            self.entry.configure(show=self.__show)
        return self.entry

    def getresult(self):
        return self.entry.get()

class DirectoryDialgoBox(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "show" in kw:
            self.__show = kw["show"]
            del kw["show"]
        else:
            self.__show = None
        if "mustexist" in kw:
            self.__mustexist = kw["mustexist"]
            del kw["mustexist"]
        else:
            self.__mustexist = None
        if "init_val" in kw:
            self.__init_val = kw["init_val"]
            del kw["init_val"]
        else:
            self.__init_val = None

        self.path = ""

        simpledialog._QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        self._pathField = tk.Entry(master, width=50)
        self._pathButton = tk.Button(master, text="Browse", command=lambda:self.__askDirectory())
        
        self._pathField.grid(row=1, column=1, padx=5, sticky="WE")
        self._pathButton.grid(row=1, column=2, sticky="WE")

        return self._pathField

    def __askDirectory(self):
        self.path = filedialog.askdirectory(initialdir=self.__init_val, mustexist=self.__mustexist)
        self._pathField.delete(0, tk.END)
        self._pathField.insert(0, self.path)

    def getresult(self):
        return self._pathField.get().strip()

class FileOpenDialogBox(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "show" in kw:
            self.__show = kw["show"]
            del kw["show"]
        else:
            self.__show = None
        if "mustexist" in kw:
            self.__mustexist = kw["mustexist"]
            del kw["mustexist"]
        else:
            self.__mustexist = None
        if "init_val" in kw:
            self.__init_val = kw["init_val"]
            del kw["init_val"]
        else:
            self.__init_val = None
        if "filetypes" in kw:
            self.__filetypes = kw["filetypes"]
            del kw["filetypes"]
        else:
            self.__filetypes = None

        self.path = ""

        simpledialog._QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        self._pathField = tk.Entry(master, width=50)
        self._pathButton = tk.Button(master, text="Browse", command=lambda:self.__askFile())
        
        self._pathField.grid(row=1, column=1, padx=5, sticky="WE")
        self._pathButton.grid(row=1, column=2, sticky="WE")

        return self._pathField

    def __askFile(self):
        self.path = filedialog.askopenfilename(initialdir=self.__init_val, mustexist=self.__mustexist, filetypes=self.__filetypes)
        self._pathField.delete(0, tk.END)
        self._pathField.insert(0, self.path)

    def getresult(self):
        return self._pathField.get().strip()

class FileSaveDialogBox(simpledialog._QueryDialog):
    def __init__(self, *args, **kw):
        if "show" in kw:
            self.__show = kw["show"]
            del kw["show"]
        else:
            self.__show = None
        if "mustexist" in kw:
            self.__mustexist = kw["mustexist"]
            del kw["mustexist"]
        else:
            self.__mustexist = None
        if "init_val" in kw:
            self.__init_val = kw["init_val"]
            del kw["init_val"]
        else:
            self.__init_val = None
        if "filetypes" in kw:
            self.__filetypes = kw["filetypes"]
            del kw["filetypes"]
        else:
            self.__filetypes = None

        self.path = ""

        simpledialog._QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        self._pathField = tk.Entry(master, width=50)
        self._pathButton = tk.Button(master, text="Browse", command=lambda:self.__askFile())
        
        self._pathField.grid(row=1, column=0, padx=5, sticky="WE")
        self._pathButton.grid(row=1, column=1, sticky="WE")

        return self._pathField

    def __askFile(self):
        self.path = filedialog.asksaveasfilename(initialdir=self.__init_val, mustexist=self.__mustexist, filetypes=self.__filetypes)
        self._pathField.delete(0, tk.END)
        self._pathField.insert(0, self.path)

    def getresult(self):
        return self._pathField.get().strip()