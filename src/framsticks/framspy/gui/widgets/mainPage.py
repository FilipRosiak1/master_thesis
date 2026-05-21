import tkinter as tk
import tkinter.ttk as ttk
from tkinter import StringVar, simpledialog, messagebox
from gui.widgets.glFrame import AppOgl
from typing import List
from functools import partial
from gui.framsutils.FramsInterface import TreeNode, InterfaceType
from gui.widgets.mainTreeView import TreeView
from gui.widgets.listGenePoolWindow import ListGenePoolWindow
from gui.widgets.listPopulationsWindow import ListPopulationsWindow
from gui.widgets.dialogBox import DirectoryDialgoBox, FileOpenDialogBox, FileSaveDialogBox
from gui.widgets.ConsoleWindow import ConsoleWindow
from gui.widgets.importWindow import ImportWindow
from gui.widgets.propertyWindow import PropertyWindow
from gui.libInterface import LibInterface
from gui.socketInterface import SocketInterface
from gui.utils import debounce
from gui.utils import windowHideAndMaximize, windowShowAndSetGeometry
from gui.widgets.ToolTip import CreateToolTip
from time import perf_counter

class MainPage(tk.Tk):
    OPENGL_WIDTH = 720
    OPENGL_HEIGHT = 480

    SIDEBAR_WIDTH = 400
    CONTROL_HEIGHT = 50
    OPTIONS_WIDTH = 100
    STATUSBAR_HEIGHT = 20

    OFFSET_HEIGHT = 60

    OPENGL_ANIMATE_DELAY = 1

    MENU_CONNECT_TO_SERVER = "Connect to server"
    MENU_CONNECT_TO_LIB = "Connect to library"

    WORKAROUND_TKINTER_FREEZE_BUG = True # There is a bug in tkinter that freezes whole app when dialogs are called too fast, hint: https://stackoverflow.com/questions/40666956/tkinter-hangs-on-rapidly-repeated-dialog

    refresh_rate_dict = {"0.1s": 100, "0.2s": 200, "0.5s": 500, "1s": 1000, "2s": 2000, "5s": 5000, "10s": 10000}

    #paths which can reload world and tree
    reload_path = ["/Experiment", "/Advanced scripting", "/World", "/User scripts"]

    def __init__(self, parent, networkAddress: str = None, libPath: str = None):
        super().__init__(parent)
        self.parent = parent
        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.title("Framsticks GUI for library/server")
        self.option_add('*tearOff', tk.FALSE)

        self.listRefreshRate = 1000
        self.frams = None
        self.canStep = False    #disable step while drawing 

        #OPENGL FRAME
        self.frame_opengl = AppOgl(self, width=self.OPENGL_WIDTH, height=self.OPENGL_HEIGHT)
        self.frame_opengl.animate = self.OPENGL_ANIMATE_DELAY
        self.frame_opengl.bind("<Configure>", self.frame_opengl.onResize)
        self.frame_opengl.bind("<Motion>", self.frame_opengl.onMouseMotion)
        self.frame_opengl.bind("<MouseWheel>", self.frame_opengl.onScroll)
        self.frame_opengl.bind("<Button>", self.frame_opengl.onMouseClick)
        self.frame_opengl.bind("<ButtonRelease>", self.frame_opengl.onMouseRelease)
        self.frame_opengl.bind("<Enter>", self.frame_opengl.onMouseEnter)

        #SIDE FRAME
        frame_sidebar = tk.Frame(master=self)
        frame_sidebar.rowconfigure(0, weight=0)
        frame_sidebar.rowconfigure(1, weight=1)
        frame_sidebar.columnconfigure(0, weight=1)

        ##CONTROL PANEL
        frame_control_panel = tk.Frame(master=frame_sidebar, width=self.SIDEBAR_WIDTH, height=self.CONTROL_HEIGHT)
        frame_control_panel.columnconfigure(0, weight=1, minsize=0)
        frame_control_panel.columnconfigure(1, weight=1, minsize=0)
        frame_control_panel.columnconfigure(2, weight=1, minsize=0)
        frame_control_panel.columnconfigure(3, weight=1, minsize=0)
        frame_control_panel.rowconfigure(0, weight=1, minsize=0)
        frame_control_panel.grid_propagate(0)

        frame_control_panel_combobox = tk.Frame(master=frame_control_panel, width=int(self.SIDEBAR_WIDTH/4))
        frame_control_panel_combobox.rowconfigure(0, weight=1, minsize=0)
        frame_control_panel_combobox.rowconfigure(1, weight=1, minsize=0)
        frame_control_panel_combobox.columnconfigure(0, weight=1, minsize=0)
        frame_control_panel_combobox.grid_propagate(0)
        self.combobox_control_panel_fps = ttk.Combobox(master=frame_control_panel_combobox, state="readonly")
        self.combobox_control_panel_fps.bind("<<ComboboxSelected>>", self.FPSCbCallback)
        self.combobox_control_panel_refresh_rate = ttk.Combobox(master=frame_control_panel_combobox, values=list(self.refresh_rate_dict.keys()), state="readonly")
        self.combobox_control_panel_refresh_rate.set(next(k for k, v in self.refresh_rate_dict.items() if v == self.listRefreshRate))
        self.combobox_control_panel_refresh_rate.bind("<<ComboboxSelected>>", self.refreshRateCbCallback)
        CreateToolTip(self.combobox_control_panel_fps, "Simulation steps to show")
        CreateToolTip(self.combobox_control_panel_refresh_rate, "Refresh rate of gene pools and populations windows")

        frame_control_panel_buttons = tk.Frame(master=frame_control_panel)
        frame_control_panel_buttons.columnconfigure(0, weight=1, minsize=0)
        frame_control_panel_buttons.columnconfigure(1, weight=1, minsize=0)
        frame_control_panel_buttons.columnconfigure(2, weight=1, minsize=0)
        frame_control_panel_buttons.rowconfigure(0, weight=1)
        self.button_control_panel_start = tk.Button(master=frame_control_panel_buttons, text="start", command=self.controlPanelStartCommand)
        self.button_control_panel_stop = tk.Button(master=frame_control_panel_buttons, text="stop", command=self.controlPanelStopCommand)
        self.button_control_panel_step = tk.Button(master=frame_control_panel_buttons, text="step", command=self.controlPanelStepCommand)
        self.button_control_panel_start["state"] = tk.DISABLED
        self.button_control_panel_stop["state"] = tk.DISABLED
        self.button_control_panel_step["state"] = tk.DISABLED
        self.button_control_panel_start.grid(row=0, column=0, sticky="NSEW")
        self.button_control_panel_stop.grid(row=0, column=1, sticky="NSEW")
        self.button_control_panel_step.grid(row=0, column=2, sticky="NSEW")
        self.combobox_control_panel_fps.grid(row=0, column=0, sticky="NSEW")
        self.combobox_control_panel_refresh_rate.grid(row=1, column=0, sticky="NSEW")
        frame_control_panel_combobox.grid(row=0, column=0, sticky="NSEW")
        frame_control_panel_buttons.grid(row=0, column=1, columnspan=3, sticky="NSEW")
        frame_control_panel.grid(row=0, column=0, sticky="NSEW")

        ##TREEVIEW
        frame_treeview = tk.Frame(master=frame_sidebar, width=self.SIDEBAR_WIDTH, height=self.OPENGL_HEIGHT - self.CONTROL_HEIGHT)
        frame_treeview.columnconfigure(0, weight=1)
        frame_treeview.columnconfigure(1, weight=0)
        frame_treeview.rowconfigure(0, weight=1)
        frame_treeview.rowconfigure(1, weight=0)

        self.treeview_treeview = TreeView(master=frame_treeview, iconPath="gui/res/icons/", selectmode="browse")
        scrollbar_treeview = ttk.Scrollbar(master=frame_treeview, orient=tk.VERTICAL, command=self.treeview_treeview.yview)
        self.treeview_treeview.configure(yscrollcommand=scrollbar_treeview.set)
        self.treeview_treeview.bind("<Double-1>", self.onTreeViewDoubleClick)
        button_treeviewRefresh = tk.Button(master=frame_treeview, text="Refresh", command=self.refreshInfoTreeCommand)

        self.treeview_treeview.grid(row=0, column=0, sticky="NSEW")
        scrollbar_treeview.grid(row=0, column=1, sticky="NSEW")
        button_treeviewRefresh.grid(row=1, column=0, sticky="NSEW")
        frame_treeview.grid(row=1, column=0, sticky="NSEW")

        #STATUS BAR
        self.motd_text = StringVar("")
        label_statusbar_motd = tk.Label(self, textvariable=self.motd_text, bd=1, height=1, relief=tk.SUNKEN, anchor=tk.W)

        #MENU BAR
        menu = tk.Menu(self)
        self.menu_open = tk.Menu(menu, tearoff=0)
        self.menu_open.add_command(label=self.MENU_CONNECT_TO_SERVER, command=self.menuConnectServerCommand)
        self.menu_open.add_command(label=self.MENU_CONNECT_TO_LIB, command=self.menuConnectLibCommand)
        self.menu_open.add_command(label="Disconnect", command=self.menuDisconnectCommand)
        self.menu_open.entryconfig("Disconnect", state="disabled")
        self.menu_open.add_separator()
        self.menu_open.add_command(label="Exit", command=self.menuExitCommand)
        menu.add_cascade(label="Main", menu=self.menu_open)
        self.menu_file = tk.Menu(menu, tearoff=0)
        self.menu_file.add_command(label="Load", command=self.menuFileLoadCommand, state="disabled")
        self.menu_file.add_command(label="Import", command=self.menuFileImportCommand, state="disabled")
        self.menu_file.add_command(label="Save experiment state as...", command=self.menuFileSaveESCommand, state="disabled")
        self.menu_file.add_command(label="Save genotypes as...", command=self.menuFileSaveGCommand, state="disabled")
        self.menu_file.add_command(label="Save simulator parameters as...", command=self.menuFileSaveSPCommand, state="disabled")
        menu.add_cascade(label="File", menu=self.menu_file)
        self.menu_options = tk.Menu(menu, tearoff=0)
        self.menu_options.add_command(label="Console", command=self.menuConsoleCommand, state="disabled")
        self.menu_options.add_command(label="Refresh world", command=self.refreshWorld, state="disabled")
        self.enableColorsVar = tk.BooleanVar(value=False)
        self.enableColorsVar.trace_add("write", self.menuEnableColorsCommand)
        self.menu_options.add_checkbutton(label="Enable colors", onvalue=True, offvalue=False, variable=self.enableColorsVar)
        self.menu_options.add_command(label="Restore windows", command=self.menuRestoreWindowsCommand)
        menu.add_cascade(label="Options", menu=self.menu_options)
        self.config(menu=menu)

        #WINDOW
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.frame_opengl.grid(row=0, column=0, sticky="NSEW")
        frame_sidebar.grid(row=0, column=1, sticky="NSEW")
        label_statusbar_motd.grid(row=1, column=0, columnspan=2, sticky="NSEW")

        #ORGANIZE WINDOWS POSITIONS
        ## need to do some workaround to determine screen width and height
        windowHideAndMaximize(self)
        maxHeight = self.winfo_rooty() + self.winfo_height()
        maxWidth = self.winfo_rootx() + self.winfo_width()
        self.rootx = self.winfo_rootx()
        windowShowAndSetGeometry(self, "%dx%d+%d+%d" % (self.OPENGL_WIDTH + self.SIDEBAR_WIDTH, self.OPENGL_HEIGHT, self.rootx, 0))
        height = self.winfo_rooty() + self.winfo_height() - self.winfo_y()

        self.list_gene_pool_window_height = int((maxHeight - height) / 2)
        self.list_gene_pool_window_pos_y = height
        self.list_gene_pool_window = ListGenePoolWindow(self, self.rootx, self.list_gene_pool_window_pos_y, self.list_gene_pool_window_height, self.listRefreshRate, self.frame_opengl.read_creature_semaphore, self.frams)
        height2 = self.list_gene_pool_window.winfo_rooty() + self.list_gene_pool_window.winfo_height() - self.list_gene_pool_window.winfo_y()
        self.list_populations_window_height = int((maxHeight - height) / 2)
        self.list_populations_window_pos_y = height + height2
        self.list_populations_window = ListPopulationsWindow(self, self.rootx, self.list_populations_window_pos_y, self.list_populations_window_height, self.listRefreshRate, self.frame_opengl.read_creature_semaphore, self.frams)

        self.bind("<FocusIn>", self.on_focus_in)
        self.bind("<FocusOut>", self.on_focus_out)

        self.fps = 50
        self.c_steps = 1

        if networkAddress and not libPath:
            self.menuConnectServerCommand(networkAddress)
        elif libPath and not networkAddress:
            self.menuConnectLibCommand(libPath)

        self.sock_adr = "127.0.0.1:9009"
        self.lib_adr = "D:\\Framsticks50rc25\\"

        self.FPSCbCallback(None)

        #step counter
        self.nb_frames = 0
        self.c_time = perf_counter()
        self.sps = 0

    def _dismiss(self):
        """dismiss main window, close all connections, release all resources."""
        if self.frams:
            self.frams.disconnect()
            
        self.destroy()

    def on_focus_in(self, event):
        """restart rendering on focusing on main window."""
        self.frame_opengl.animate = self.OPENGL_ANIMATE_DELAY
        self.frame_opengl.tkExpose(None)

    def on_focus_out(self, event):
        """stop the rendering when main window lose focus."""
        self.frame_opengl.animate = 0

    def menuConnectServerCommand(self, adr: str = None):
        """on "connect to server" button command."""
        #if connection started from command argument
        if adr:
            address = adr
        #else ask for server address
        else:
            address = simpledialog.askstring("Server address", "Address", parent=self, initialvalue=self.sock_adr)
        if address:
            ip, port = address.split(":")
            try:
                self.frams = SocketInterface()
                self.frams.registerRunningChangeEventCallback(self.refreshControlPanelButtons)
                self.frams.registerTreeviewRefreshEventCallback(self.refreshInfoTree)
                self.frams.connect(ip, int(port)) #try to connect to server
                self.sock_adr = address
            except ConnectionError: #if connection cannot be established
                messagebox.showerror(message="Cannot connect to server")
                if self.frams:
                    self.frams.disconnect()
                self.frams = None
        
        #if connected successfully 
        if self.frams and self.frams.frams.comm.connected:
            self._connect(address)

    def menuConnectLibCommand(self, path: str = None):
        """on "connect to library" button command."""
        #if connection started from command argument
        if path:
            address = path
        #else ask for library path
        else:
            address = DirectoryDialgoBox("Framsticks library path", "Path", parent=self, initialvalue=self.lib_adr)
            address = address.result
        if address:
            try:
                self.frams = LibInterface()
                self.frame_opengl.onDraw = self.onDraw
                self.prev_run = False
                self.frams.connect(address, 0)
                self.lib_adr = address
            except ConnectionError:
                messagebox.showerror(message="Cannot find Framsticks library")

        self._connect(address)

    def _connect(self, address):
        """set all control's states if connected."""
        if address and self.frams:
            #set creatures read callback
            self.frame_opengl.frams_readCreatures = self.frams.readCreatures
            #prepare populations and gene pools windows
            self.list_populations_window.frams = self.frams
            self.list_gene_pool_window.frams = self.frams
            self.list_populations_window.refresh = True
            self.list_gene_pool_window.refresh = True
            self.list_populations_window.refreshPopulations()
            self.list_gene_pool_window.refreshGenePools()
            #enable control buttons
            self.button_control_panel_start["state"] = tk.NORMAL
            self.button_control_panel_stop["state"] = tk.NORMAL
            self.button_control_panel_step["state"] = tk.NORMAL
            self.refreshInfoTree()
            self.motd_text.set(self.frams.getMotd())
            #setup all menus
            self.menu_open.entryconfig("Disconnect", state="normal")
            self.menu_open.entryconfig(self.MENU_CONNECT_TO_SERVER, state="disabled")
            self.menu_open.entryconfig(self.MENU_CONNECT_TO_LIB, state="disabled")
            self.menu_file.entryconfig("Load", state="normal")
            self.menu_file.entryconfig("Import", state="normal")
            self.menu_file.entryconfig("Save experiment state as...", state="normal")
            self.menu_file.entryconfig("Save genotypes as...", state="normal")
            self.menu_file.entryconfig("Save simulator parameters as...", state="normal")
            self.menu_options.entryconfig("Refresh world", state="normal")

            if self.frams.interfaceType == InterfaceType.SOCKET:
                self.menu_options.entryconfig("Console", state="normal")
            else:
                self.menu_options.entryconfig("Console", state="disabled")

            self.fps_values = self.frams.getFPSDefinitions()
            def mapper(fps, step):
                return "Every{}".format(", {} fps".format(fps) if fps > 0 else "") if step == 1 else "1:{}".format(step)
            self.fps_mapvalues = [mapper(fps, step) for fps, step in self.fps_values]
            init_val = mapper(self.fps, self.c_steps)
            self.combobox_control_panel_fps['values'] = self.fps_mapvalues
            self.combobox_control_panel_fps.set(init_val)

            self.FPSCbCallback(None)

            self.refreshWorld()
            self.canStep = True     #enable step while drawing 

    def menuDisconnectCommand(self):
        """set all control's states if disconnected."""
        if self.frams:
            self.canStep = False
            self.frams.disconnect()
            self.button_control_panel_start["state"] = tk.DISABLED
            self.button_control_panel_stop["state"] = tk.DISABLED
            self.button_control_panel_step["state"] = tk.DISABLED
            self.list_populations_window.refresh = False
            self.list_gene_pool_window.refresh = False  
            self.list_populations_window.clearList()
            self.list_gene_pool_window.clearList()
            self.frame_opengl.frams_readCreatures = None
            self.frame_opengl.onDraw = lambda: None
            self.refreshInfoTree()
            self.frams = None
            self.frame_opengl.swap_buffer.clear()
            self.motd_text.set("")
            self.menu_open.entryconfig("Disconnect", state="disabled")
            self.menu_open.entryconfig(self.MENU_CONNECT_TO_SERVER, state="normal")
            self.menu_open.entryconfig(self.MENU_CONNECT_TO_LIB, state="normal")
            self.menu_options.entryconfig("Console", state="disabled")
            self.menu_file.entryconfig("Load", state="disabled")
            self.menu_file.entryconfig("Import", state="disabled")
            self.menu_file.entryconfig("Save experiment state as...", state="disabled")
            self.menu_file.entryconfig("Save genotypes as...", state="disabled")
            self.menu_file.entryconfig("Save simulator parameters as...", state="disabled")
            self.menu_options.entryconfig("Refresh world", state="disabled")

    def menuExitCommand(self):
        self._dismiss()

    def FPSCbCallback(self, event):
        """handle fps change."""
        value = self.combobox_control_panel_fps.get()
        if value:
            i = self.fps_mapvalues.index(value)
            fps, step = self.fps_values[i]
            self.fps = fps
            self.c_steps = step

            if fps == -1:
                self.frame_opengl.animate = self.OPENGL_ANIMATE_DELAY = 1
                self.frame_opengl.REFRESH_RATE = 0.001 #set to "as fast as possible" because how often redrawing is called is decided by the timer
            else:
                self.frame_opengl.animate = self.OPENGL_ANIMATE_DELAY = int(1000 / fps)
                self.frame_opengl.REFRESH_RATE = 1 / fps

    def refreshRateCbCallback(self, event):
        """handle change of refresh rate of gene pools and populations windows."""
        value = self.combobox_control_panel_refresh_rate.get()
        if value:
            self.listRefreshRate = self.refresh_rate_dict[value]
            self.list_gene_pool_window.refreshRate = self.listRefreshRate
            self.list_populations_window.refreshRate = self.listRefreshRate

    def controlPanelStartCommand(self):
        if self.frams:
            self.frams.start()

    def controlPanelStopCommand(self):
        if self.frams:
            self.frams.stop()
            self.motd_text.set("")

    def refreshControlPanelButtons(self, is_running: bool):
        if is_running:
            self.button_control_panel_start["state"] = tk.DISABLED
            self.button_control_panel_stop["state"] = tk.NORMAL
            self.button_control_panel_step["state"] = tk.DISABLED
        else:
            self.button_control_panel_start["state"] = tk.NORMAL
            self.button_control_panel_stop["state"] = tk.DISABLED
            self.button_control_panel_step["state"] = tk.NORMAL

    def controlPanelStepCommand(self):
        """hangle step button."""
        if self.frams:  #if connected to frams
            if self.frams.interfaceType == InterfaceType.LIB:
                self.canStep = False    #disable step while drawing 
                run = self.frams.getSimulationStatus()
                self.frams.start()
                self.frams.step()
                if not run:
                    self.frams.stop()
                self.canStep = True     #enable step while drawing 
            else:
                self.canStep = False    #disable step while drawing 
                self.frams.step()
                self.canStep = True     #enable step while drawing 

    def refreshInfoTreeCommand(self):
        self.refreshInfoTree()
        self.refreshWorld()

    @debounce(1)
    def refreshInfoTree(self):
        """refresh info tree
            debounce decorator prevents refreshing too often."""
        self.treeview_treeview.delete(*self.treeview_treeview.get_children())
        if self.frams:
            tree: List[TreeNode] = self.frams.makeInfoTree()
            self._recRefreshInfoTree(tree, True)

    def _recRefreshInfoTree(self, node: TreeNode, open: bool = False):
        #self.treeview_treeview.insert("" if not node.parent else node.parent.node.Id, index="end", iid=node.node.Id, text=node.node.Name)
        self.treeview_treeview.insert(self._generateInfoTreeParent(node), index="end", iid=self._generateInfoTreeId(node), text=node.node.p["name"], ico=node.node.p["id"], open=open)
        for child in node.children:
            self._recRefreshInfoTree(child)

    def _generateInfoTreeParent(self, node: TreeNode):
        if not node.parent:
            return ""
        return self._generateInfoTreeId(node.parent)

    def _generateInfoTreeId(self, node: TreeNode):
        """generate unique id for info tree using paths and ids."""
        response = node.node.p["id"]
        while node.parent:
            if node.parent:
                if node.parent.node.p["id"] != '/':
                    response = node.parent.node.p["id"] + "/" + response
                else:
                    response = node.parent.node.p["id"] + response
                node = node.parent
        return response

    def onTreeViewDoubleClick(self, event):
        if self.frams:
            item = self.treeview_treeview.identify_row(event.y)
            if item:
                info = partial(self.frams.readParameterDetails, item)
                update = partial(self.frams.writeParameterDetail, item)

                if any(item.lower().endswith(x.lower()) for x in self.reload_path):
                    pw = PropertyWindow(self, item, self.rootx, info, update, self.frams.getError, self.frams, self.frame_opengl.read_creature_semaphore, self.refreshWorld)
                else:
                    pw = PropertyWindow(self, item, self.rootx, info, update, self.frams.getError, self.frams, self.frame_opengl.read_creature_semaphore)
                return 'break'

    def onDraw(self):
        """on draw callback, only for frams library."""
        if self.frams:
            run = self.frams.getSimulationStatus()
            if run != self.prev_run:    #check if simulation status changed since last onDraw
                self.refreshControlPanelButtons(run)    #if so, refresh command buttons
                self.prev_run = run
            if run:
                if self.canStep: #if running and can step, perform N steps 
                    with self.frame_opengl.read_creature_semaphore:
                        for i in range(self.c_steps):
                            self.frams.step()

                            #calculate steps per second
                            c_time = perf_counter()
                            self.nb_frames += 1
                            e_time = c_time - self.c_time
                            if e_time >= 1.0:
                                self.sps = int(self.nb_frames / e_time)
                                self.c_time = perf_counter()
                                self.nb_frames = 0
                                self.motd_text.set("{} steps/second".format(self.sps))
            error = self.frams.getError()   #check for errors, not used anymore
            if error:
                messagebox.showerror("Framsticks error", error)

    def menuConsoleCommand(self):
        if self.frams.interfaceType == InterfaceType.SOCKET:
            ConsoleWindow(self, self.frams.frams)
        else:
            self.menu_options.entryconfig("Console", state="disabled")

    def menuFileLoadCommand(self):
        if self.frams:
            self.canStep = False
            path = FileOpenDialogBox("Open File", "Path", parent=self, initialvalue=self.lib_adr, filetypes=[("Framsticks files", ['*.expt','*.gen','*.sim']), ("Experiment state", "*.expt"), ("Genotypes", "*.gen"), ("Simulator parameters", "*.sim"), ("All files", "*.*")])
            path = path.result
            if path:
                #acquire read creature semaphore, so render thread could not ask for them, while they are changing
                with self.frame_opengl.read_creature_semaphore:
                    self.frams.loadFile(path)
                self.refreshWorld()
            self.canStep = True

    def menuFileImportCommand(self):
        if self.frams:
            self.canStep = False
            path = FileOpenDialogBox("Open File", "Path", parent=self, init_val=self.lib_adr, filetypes=[("Framsticks files", ['*.expt','*.gen','*.sim']), ("Experiment state", "*.expt"), ("Genotypes", "*.gen"), ("Simulator parameters", "*.sim"), ("All files", "*.*")])
            path = path.result
            if path:
                #acquire read creature semaphore, so render thread could not ask for them, while they are changing
                with self.frame_opengl.read_creature_semaphore:
                    def handle_import_options():
                            options = ImportWindow(parent=self, filename=path)
                            options = options.result
                            self.frams.importFile(path, options)
                    if self.WORKAROUND_TKINTER_FREEZE_BUG:
                        self.after(1, handle_import_options) 
                    else:
                        handle_import_options()
                    
            self.canStep = True

    def menuFileSaveESCommand(self):
        if self.frams:
            path = FileSaveDialogBox("Save experiment state to:", "Path", parent=self, initialvalue=self.lib_adr, filetypes=[("Experiment state", "*.expt")])
            path = path.result
            if path:
                if path.split(".")[-1] != "expt":
                    path += ".expt"
                self.frams.saveFile(path, 1)

    def menuFileSaveGCommand(self):
        if self.frams:
            path = FileSaveDialogBox("Save genotypes to:", "Path", parent=self, initialvalue=self.lib_adr, filetypes=[("Genotypes", "*.gen")])
            path = path.result
            if path:
                if path.split(".")[-1] != "gen":
                    path += ".gen"
                self.frams.saveFile(path, 2)

    def menuFileSaveSPCommand(self):
        if self.frams:
            path = FileSaveDialogBox("Save simulator parameters to:", "Path", parent=self, initialvalue=self.lib_adr, filetypes=[("Simulator parameters", "*.sim")])
            path = path.result
            if path:
                if path.split(".")[-1] != "sim":
                    path += ".sim"
                self.frams.saveFile(path, 4)

    def refreshWorld(self):
        """refresh world parameters and shape."""
        worldType = self.frams.getWorldType()
        worldSize = self.frams.getWorldSize()
        worldWaterLevel = self.frams.getWorldWaterLevel()
        worldBoundaries = self.frams.getWorldBoundaries()
        worldMap = self.frams.getWorldMap()
        simType = self.frams.getSimtype()
        self.frame_opengl.reloadWorld(worldType, simType, worldSize, worldMap, worldBoundaries, worldWaterLevel)

    def menuEnableColorsCommand(self, *_):
        v = self.enableColorsVar.get()
        self.frame_opengl.frams_readCreatures_color = v

    def menuRestoreWindowsCommand(self):
        self.geometry("%dx%d+%d+%d" % (self.OPENGL_WIDTH + self.SIDEBAR_WIDTH, self.OPENGL_HEIGHT, self.rootx, 0))

        if self.list_gene_pool_window.opened:
            self.list_gene_pool_window._dismiss()
        self.list_gene_pool_window = ListGenePoolWindow(self, self.rootx, self.list_gene_pool_window_pos_y, self.list_gene_pool_window_height, self.listRefreshRate, self.frame_opengl.read_creature_semaphore, self.frams)
        if self.frams:
            self.list_gene_pool_window.frams = self.frams
            self.list_gene_pool_window.refresh = True
            self.list_gene_pool_window.refreshGenePools()

        if self.list_populations_window.opened:
            self.list_populations_window._dismiss()
        self.list_populations_window = ListPopulationsWindow(self, self.rootx, self.list_populations_window_pos_y, self.list_populations_window_height, self.listRefreshRate, self.frame_opengl.read_creature_semaphore, self.frams)
        if self.frams:
            self.list_populations_window.frams = self.frams
            self.list_populations_window.refresh = True
            self.list_populations_window.refreshPopulations()
