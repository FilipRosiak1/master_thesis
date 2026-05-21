from threading import Timer
import os

class Swap:
    def __init__(self, init = None) -> None:
        self.init = init
        self.clear()
        self.select = False
        self.empty = False

    def update(self, new):
        if not new:
            if self.empty == False:
                self.empty = True
                return
        else:
            self.empty = False

        if self.select:
            self.one = new
        else:
            self.two = new
        self.select = not self.select

    def get(self):
        if self.select:
            return self.two
        else:
            return self.one

    def clear(self):
        self.one = self.init
        self.two = self.init


#source: https://gist.github.com/walkermatt/2871026
def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after wait seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)
            try:
                debounced.t.cancel()
            except(AttributeError):
                pass
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
        return debounced
    return decorator


# https://stackoverflow.com/questions/39058038/wm-attributes-and-zoomed-doesnt-work
# https://stackoverflow.com/questions/18394597/is-there-a-way-to-create-transparent-windows-with-tkinter
def windowHideAndMaximize(wnd): # to get the size of working area on screen (screen minus taskbars, toolbars etc.) - make invisible maximized window
    if os.name=='posix':
        wnd.wait_visibility()
    wnd.attributes("-alpha", 0)
    if os.name=='posix':
        wnd.attributes("-zoomed", 1)
    else:
        wnd.state('zoomed')
    wnd.update()

def windowShowAndSetGeometry(wnd, geometry):
    if os.name=='posix':
        wnd.attributes("-zoomed", 0)
        wnd.wait_visibility()
    else:
        wnd.state('normal')
    wnd.update()
    wnd.geometry(geometry)
    wnd.attributes("-alpha", 1)
    wnd.update()
