import tkinter as tk
import tkinter.ttk as ttk
import glob
from pathlib import Path
from PIL import Image, ImageTk

class TreeView(ttk.Treeview):
    ICON_SIZE = 32
    def __init__(self, master=None, iconPath="", **kw) -> None:
        kw["show"] = "tree"
        super().__init__(master=master, **kw)
        self.icons = {}
        self.s = ttk.Style()
        self.s.configure('Treeview', rowheight=self.ICON_SIZE)
        self.iconLoader(iconPath)

    def iconLoader(self, path):
        files = glob.glob(path + "\\*.png")
        for file in files:
            icon = Image.open(file)
            icon = icon.resize((self.ICON_SIZE, self.ICON_SIZE))
            name = Path(file).name
            name = name[:name.index('.')]
            self.icons[name.lower()] = ImageTk.PhotoImage(icon)

    def insert(self, parent, index, iid=None, **kw) -> str:
        if "ico" in kw:
            ico = kw["ico"].lower()
            del kw["ico"]
        else:
            ico = None

        if ico in self.icons:
            img = self.icons[ico]
            return super().insert(parent, index, iid=iid, image=img, **kw)
        return super().insert(parent, index, iid=iid, **kw)