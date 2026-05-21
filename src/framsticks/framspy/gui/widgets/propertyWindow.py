import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
from typing import Dict, List, Callable, Tuple
from gui.framsutils.framsProperty import Property, PropertyCallback, propertyToTkinter
from gui.framsutils.FramsInterface import FramsInterface, InterfaceType
from gui.widgets.ToolTip import CreateToolTip
from gui.utils import windowHideAndMaximize, windowShowAndSetGeometry

'''
BUG: when you open this window during the creature loading for render (commThread in glFrame) and modify the list of creatures (with buttons) or parts (with editing genotype), it can get stuck on the semaphore or throw an exception when exiting the window and returning to creature loading.
SOLUTION: open window only when the semaphore is released.
'''
class PropertyWindow(tk.Toplevel):
    def __init__(self, parent, title: str, posX, dataCallback: Callable[[None], List[Property]], propUpdateCallback: Callable[[str, str], None], getError: Callable[[None], str or None], frams: FramsInterface, semaphore = None, onUpdate: Callable[[None], None] = None):
        super().__init__(parent)
        self.parent = parent
        #self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.title(title)

        self.propUpdateCallback: Callable[[str, str], None] = propUpdateCallback
        self.dataCallback: Callable[[None], List[Property]] = dataCallback
        self.getError: Callable([None], str or None) = getError
        self.onUpdate: Callable([None], None) = onUpdate
        self.frams: FramsInterface = frams

        #MAIN SECTION
        frame_main = tk.Frame(master=self)
        notebook_notebook = ttk.Notebook(frame_main)
        frames: Dict[str, tk.Frame] = {}
        frames_idx: Dict[str, int] = {}
        data = self.dataCallback()

        self.callbacks: List[PropertyCallback] = []
        self.semaphore = semaphore

        for prop in data:
            if "group" not in prop.p:
                prop.p["group"] = "other"
                
            if prop.p["group"] not in frames:
                frames[prop.p["group"]] = tk.Frame(master=frame_main)
                frames_idx[prop.p["group"]] = 0

                frames[prop.p["group"]].columnconfigure(0, weight=0, minsize=0)
                frames[prop.p["group"]].columnconfigure(1, weight=1, minsize=0)
                notebook_notebook.add(frames[prop.p["group"]], text=prop.p["group"])

            widget, callback = propertyToTkinter(prop, frames[prop.p["group"]])
            if widget:
                label = tk.Label(master=frames[prop.p["group"]], text=prop.p["name"], anchor='w')
                if prop.p["type"][0] == 'p':
                    label.text = ""
                label.grid(row=frames_idx[prop.p["group"]], column=0, sticky="NSEW")
                if issubclass(type(widget), tk.Checkbutton):
                    widget.grid(row=frames_idx[prop.p["group"]], column=1, sticky="NSW")
                else:
                    widget.grid(row=frames_idx[prop.p["group"]], column=1, sticky="NSEW")
                if callback:
                    self.callbacks.append(callback)
                if prop.p["help"] != "":
                    CreateToolTip(label, prop.p["help"])
                    CreateToolTip(widget, prop.p["help"])
                if issubclass(type(widget), tk.Text) and prop.p["type"][0] == 's':
                    frames[prop.p["group"]].rowconfigure(frames_idx[prop.p["group"]], weight=1, minsize=0)
                else:
                    frames[prop.p["group"]].rowconfigure(frames_idx[prop.p["group"]], weight=0, minsize=0)
                frames_idx[prop.p["group"]] += 1

        notebook_notebook.grid(row=0, column=0, sticky="NSEW") #sometimes it freezes here
        frame_main.columnconfigure(0, weight=1, minsize=0)
        frame_main.rowconfigure(0, weight=1, minsize=0)

        #CONTROL BUTTONS
        frame_buttons = tk.Frame(master=frame_main)
        self.button_refresh = tk.Button(master=frame_buttons, text="Refresh", command=self.refreshButtonCommand)
        self.button_cancel = tk.Button(master=frame_buttons, text="Cancel", command=self.cancelButtonCommand)
        self.button_apply = tk.Button(master=frame_buttons, text="Apply", command=self.applyButtonCommand)
        self.button_ok = tk.Button(master=frame_buttons, text="OK", command=self.okButtonCommand)
        self.button_refresh.grid(row=0, column=0, sticky="E")
        self.button_cancel.grid(row=0, column=1, sticky="E")
        self.button_apply.grid(row=0, column=2, sticky="E")
        self.button_ok.grid(row=0, column=3, sticky="E")

        frame_buttons.columnconfigure(0, weight=1, minsize=0)
        frame_buttons.columnconfigure(1, weight=1, minsize=0)
        frame_buttons.columnconfigure(2, weight=1, minsize=0)
        frame_buttons.columnconfigure(3, weight=1, minsize=0)
        frame_buttons.rowconfigure(0, weight=1, minsize=0)
        frame_buttons.grid(row=1, column=0, sticky="SE")
        frame_main.rowconfigure(1, weight=0, minsize=0)

        frame_main.grid(row=0, column=0, sticky="NSEW")

        self.columnconfigure(0, weight=1, minsize=0)
        self.rowconfigure(0, weight=1, minsize=0)

        def parseGeometryString(geometry: str) -> Tuple[int, int, int, int]:
            widthxrest = geometry.split('x')
            heightxy = widthxrest[1].split('+')
            maxwidth = int(widthxrest[0])
            maxheight = int(heightxy[0])
            x = int(heightxy[1])
            y = int(heightxy[2])
            return maxwidth, maxheight, x, y

        self.update_idletasks()

        width, height, x, y = parseGeometryString(self.winfo_geometry())

        windowHideAndMaximize(self)
        rootx = self.winfo_rootx()

        maxWidth, maxHeight, x, y = parseGeometryString(self.winfo_geometry())
        x = (maxWidth - width) / 2
        y = (maxHeight - height) / 2

        windowShowAndSetGeometry(self, "+%d+%d" % (int(rootx + x), int(y)))
        self.maxsize(maxWidth, maxHeight)

        self.grab_set()

    def refreshButtonCommand(self):
        data = self.dataCallback()
        for d in data:
            prop = next((x for x in self.callbacks if d.p["id"] == x.id), None)
            if prop:
                prop.updateValue(d)

    def cancelButtonCommand(self):
        self._dismiss()

    def applyButtonCommand(self):
        if self.propUpdateCallback:
            updated = False
            with self.semaphore:
                for c in self.callbacks:
                    if c.changed():
                        self.propUpdateCallback(c.id, c.rawValue(self.frams.interfaceType == InterfaceType.SOCKET))
                        updated = True
                        error = self.getError()
                        if error:
                            messagebox.showerror("Framsticks error", error)
                self.refreshButtonCommand()
                if updated and self.onUpdate:
                    self.onUpdate()
    
    def okButtonCommand(self):
        self.applyButtonCommand()
        self.cancelButtonCommand()

    def _dismiss(self):
        self.grab_release()
        self.destroy()