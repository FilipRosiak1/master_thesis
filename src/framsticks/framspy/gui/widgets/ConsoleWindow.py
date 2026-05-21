import tkinter as tk
from gui.widgets.ScrolledText import ScrolledText
from gui.framsutils.FramsSocket import FramsSocket
from gui.utils import windowHideAndMaximize, windowShowAndSetGeometry
from typing import Tuple, Callable

class ConsoleWindow(tk.Toplevel):
    def __init__(self, master, frams: FramsSocket) -> None:
        super().__init__(master)

        self.frams: FramsSocket = frams
        self.frams.registerMessageEventCallback(self.messageCallbackEvent)

        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.title("Console")

        self.frame_main = tk.Frame(master=self)

        self.console = ScrolledText(self.frame_main, height=10, state="disabled")
        self.console.grid(row=0, column=0, columnspan=2, sticky="NSEW")

        self.input = tk.Entry(self.frame_main)
        self.input.bind("<Return>", self.buttonSendCommand)
        self.input.grid(row=1, column=0, sticky="NSEW")

        self.sendButton = tk.Button(self.frame_main, text="Send", command=self.buttonSendCommand)
        self.sendButton.grid(row=1, column=1, sticky="SE")

        self.frame_main.rowconfigure(0, weight=1)
        self.frame_main.rowconfigure(1, weight=0)

        self.frame_main.columnconfigure(0, weight=1)
        self.frame_main.columnconfigure(1, weight=0)

        self.frame_main.grid(row=0, column=0, sticky="NSEW")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

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

    def buttonSendCommand(self, event=None):
        request = self.input.get().strip()
        if len(request) > 0:
            self.sendButton.config(state=tk.DISABLED)
            self.input.delete(0, tk.END)

            response = self.frams.sendRequest(request)

            self.console.config(state=tk.NORMAL)
            self.console.insert(tk.END, ">>> {}\n".format(request))
            for r in response:
                self.console.insert(tk.END, r)
            self.console.see(tk.END)
            self.console.config(state=tk.DISABLED)

            self.sendButton.config(state=tk.NORMAL)

    def _dismiss(self):
        self.frams.registerMessageEventCallback(None)
        self.destroy()

    def messageCallbackEvent(self, block, header):
        response = self.frams._infoParser(block)

        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, ">>> {}\n".format(response[0].p["func"]))
        for r in str(response[0].p["message"]):
            self.console.insert(tk.END, r)
        self.console.insert(tk.END, "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
