# Python GUI for Framsticks library and server

## Table of contents

- [About](#about)
- [Running](#running)
- [Sources and directories](#sources-and-directories)
- [Threading](#threading)
- [Development](#development)



## About

This GUI offers basic features when connecting to a native Framsticks library (```dll/so/dylib```) or the [Framsticks server](https://www.framsticks.com/common/server.html). While there exist native GUIs for Framsticks which are much faster and offer more features, the advantage of this GUI is that it is written entirely in Python (thus being portable), it is entirely open-source, and can be freely modified or serve as an implementation example. This GUI is also completely separate from the Framsticks simulator (either the library or the server).




## Running

Run the ```gui.py``` script.




## Sources and directories

```text
framspy/
├── frams.py
├── gui.py
└── gui/
    ├── framsutils/
    ├── res/
    │   ├── icons/
    │   ├── img/
    │   └── obj/
    ├── tests/
    ├── visual/
    │   └── shaders/
    └── widgets/
```

- ```framsutils``` contains all classes used for low-level data manipulation.
- ```res``` is a resource directory.
- ```tests``` contains basic test classes.
- ```visual``` contains all OpenGL-related classes and files (shaders).
- ```widgets``` contains all tkinter widgets.




## Threading

The GUI runs the ```Tkinter``` main loop in the main thread.

```glFrame``` runs an additional thread to periodically and asynchronously obtain creatures data to render without freezing the render window. This is especially required for the socket connection, because requesting data is time-consuming relatively to rendering.

```FramsSocket``` runs the communication module with ```asyncio``` in another thread to provide fast and concurrent network communication. 

The communication module uses another thread to detect asynchronous events from server and to call an adequate callback.





## Development

### Communication

If you want to add another feature to socket communication, send feature negotiation request in ```CommWrapper```'s *start* method in ```comm.py```.

There is no generic system for registering for new events in the socket communication. If you want to register for a new event, add callback in ```EventConsumer```'s *run* method in ```comm.py``` and register for an event in ```FramsSocket```'s *initConnection* method in ```FramsSocket.py```.

### UI

All property features for input widgets are handled in ```framsProperty.py``` in the *propertyToTkinter* function. 

All widget's definitions for the main window are added in the *\_\_init\_\_* method of the ```MainPage``` class in ```mainPage.py```. This method is divided into sections:
- OPENGL FRAME
- SIDE FRAME
    - CONTROL PANEL
    - TREEVIEW
- STATUS BAR
- MENU BAR
- WINDOW
- ORGANIZE WINDOWS POSITIONS for all displayed windows

If you need to ask for more fields in ```ListGenePoolWindow``` or ```ListPopulationsWindow```, add their ids to *self.fields* list. If you want to show them as columns, make sure to add them in the right order at the begining of the list, because only *len(self.headers)* first fields with *self.headers* headers are shown. Also fill *self.headers_width* with the correct percentage value of the initial window width, for example:
```python
fields = ["name", "genotype", "numjoints", "index", "uid"]
headers = ["Name", "Genotype"]
headers_width = [0.5, 0.5]
```
The above example asks for *name*, *genotype*, *numjoints*, *index*, *uid*, but only shows *name* in the *Name* column and *genotype* in the *Genotype* column with 50% and 50% widths for each column. The remaining fields can be used during the preparation of the row.

### Contributing

There are a number of "TODO" items (features, bugs) for this Python GUI described [here](https://docs.google.com/document/d/1KeY6RceeeiPDkt_Z6S2GihpcA4YsfujNfk3AErvXP3Q/edit?usp=sharing). Contact ```support@framsticks.com``` to learn more about the development.

