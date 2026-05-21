import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from time import mktime
from gui.widgets.ScrolledText import ScrolledText
import re

trans = {
	"~": r"\~",
	"\\": r"\\",
	"\"": r"\"",
	"\n": r"\n",
	"\t": r"\t"}

def encodeString(data: str) -> str:
	return data.translate(data.maketrans(trans))

datetimeformat = '%Y-%m-%d %H:%M:%S'

class Property:
	def __init__(self, Id: str = "", Name: str = "", Type: str = "", p: Dict = None) -> None:
		self.p = {}
		self.p["id"] = Id
		self.p["name"] = Name
		self.p["type"] = Type
		self.p["flags"] = 0
		self.p["help"] = ""
		self.p["value"] = None
		if p:
			self.p = {**self.p, **p}

class PropertyCallback:
	def __init__(self, widget: tk.Widget, Id: str, xtype: str) -> None:
		self.widget: tk.Widget = widget
		self.id = Id
		self._changed = False
		self.type = xtype[0]
		self.subtype = None
		if len(xtype) > 1:
			self.subtype = xtype[1]

		if isinstance(self.widget, tk.Checkbutton):
			self.var = self.widget.variable
		elif isinstance(self.widget, tk.Text):
			self.widget.edit_modified(False)
		elif isinstance(self.widget, tk.Button):
			pass

		self.initValue = self.value()

	def changed(self) -> bool:
		if isinstance(self.widget, tk.Text):
			self._changed = self.widget.edit_modified()
		else:
			if self.initValue != self.value():
				self._changed = True
		return self._changed

	def restart(self) -> None:
		self.initValue = self.value()
		self._changed = False

	def value(self) -> str:
		if isinstance(self.widget, tk.Checkbutton):
			return self._checkButton()
		elif isinstance(self.widget, ttk.Combobox):
			return self._combobox()
		elif isinstance(self.widget, ttk.Spinbox):
			return self._spinbox()
		elif isinstance(self.widget, tk.Text):
			return encodeString(self._text())
		elif isinstance(self.widget, tk.Entry):
			return self._entry()
		return None

	def rawValue(self, translate: bool = True) -> str:
		if self.type == 'd':
			if self.subtype == 'b':
				return int(self.value(), 2)
			elif self.subtype == 'c':
				return int(self.value(), 16)
			else:
				return self.value()
		elif self.type == 'f':
			if self.subtype == 't':
				el = datetime.strptime(self.value(), datetimeformat)
				t = el.timetuple()
				return mktime(t)
			elif self.subtype == 'i':
				return float(self.value())
			else:
				return self.value()
		elif self.type == 's':
			if not translate and isinstance(self.widget, tk.Text):
				return self._text() 
			else:
				return self.value()
		elif self.type == 'p':
			return self.value()

	def updateValue(self, value: Property) -> None:
		_, cb = propertyToTkinter(value, self.widget.master)

		if isinstance(self.widget, tk.Checkbutton):
			if value.p["value"] == '1' or value.p["value"] == 1:
				self.widget.select()
			else:
				self.widget.deselect()
		elif isinstance(self.widget, ttk.Combobox):
			if hasattr(self.widget, "mapvalues"):
				self.widget.set(list(self.widget.mapvalues.keys())[list(self.widget.mapvalues.values()).index(cb.value())])
			else:
				self.widget.set(cb.value())
		elif isinstance(self.widget, ttk.Spinbox):
			self.widget.set(cb.value())
		elif isinstance(self.widget, tk.Text):
			self.widget.config(state=tk.NORMAL)
			if len(self.value()) > 0:
				self.widget.delete("1.0", tk.END)
			self.widget.insert("1.0", cb._text()) #translates and escapes \\ as expected 
			self.widget.edit_modified(False)
			self.widget.config(state=tk.DISABLED)
		elif isinstance(self.widget, tk.Entry):
			self.widget.delete(0, tk.END)
			self.widget.insert(0, cb.value())

	def _checkButton(self) -> str:
		return "1" if self.var.get() else "0"

	def _combobox(self) -> str:
		if hasattr(self.widget, "mapvalues"):
			return self.widget.mapvalues.get(self.widget.get())
		else:
			return self.widget.get()

	def _spinbox(self) -> str:
		return str(self.widget.get())

	def _text(self) -> str:
		return self.widget.get("1.0", "end").strip()

	def _entry(self) -> str:
		return self.widget.get().strip()

def propertyToTkinter(prop: Property, master: tk.Widget) -> Optional[Tuple[tk.Widget, PropertyCallback]]:
	global trans
	widget = None
	callback: PropertyCallback = None
	t = prop.p["type"].split()
	propType = t[0]

	readonly = True if int(prop.p["flags"]) & 1 or int(prop.p["flags"]) & 16 else False
	visible = False if int(prop.p["flags"]) & 32 else True

	if not visible:
		return None, None

	class MinMaxDefault:
		def __init__(self, minValue, maxValue, default = None) -> None:
			self.minValue = minValue
			self.maxValue = maxValue
			self.default = default

	def minMaxDefault(prop_type: List) -> MinMaxDefault:
		if len(prop_type) >= 4:
			return MinMaxDefault(prop_type[1], prop_type[2], prop_type[3])
		elif len(prop_type) >= 3:
			return MinMaxDefault(prop_type[1], prop_type[2])
		else:
			return MinMaxDefault(None, None)

	minmax = minMaxDefault(t)

	if propType[0] == 'd':
		value = int(prop.p["value"])
		if len(propType) > 1:
			if propType[1] == 'b':
				value = bin(value)
			elif propType[1] == 'c':
				value = hex(value)

		if minmax.minValue == '0' and minmax.maxValue == '1' and prop.p["type"].find('~') == -1:
			var = tk.BooleanVar()
			widget = tk.Checkbutton(master=master, text='', onvalue=1, offvalue=0, var=var)
			if value == 1:
				widget.select()
			if readonly:
				widget['state'] = 'disabled'
			widget.variable = var
		elif prop.p["type"].find('~') > 0:
			options = prop.p["type"].split('~')[1:]
			widget = ttk.Combobox(master=master, values=options, state="readonly")
			widget.mapvalues = {k:v for (v, k) in enumerate(options, int(t[1]))}
			widget.set(options[value])
		else:
			minVal = 0 if minmax.minValue is None else minmax.minValue
			maxVal = 100 if minmax.maxValue is None else minmax.maxValue
			widget = ttk.Spinbox(master=master, from_=minVal, to=maxVal)
			widget.set(value)
			if readonly:
				widget["state"] = "disabled"
	elif propType[0] == 'f':
		value = None
		var = None
		if len(propType) > 1:
			if propType[1] == 't':
				timestamp = float(prop.p["value"])
				value = datetime.fromtimestamp(timestamp).strftime(datetimeformat)
			elif propType[1] == 'i':
				var = tk.StringVar()
				value = float(prop.p["value"])
		else:
			var = tk.StringVar()
			value = float(prop.p["value"])

		if var:
			def callback(*_):
				val = var.get()
				try:
					v = float(val)
					if minmax.minValue and v < float(minmax.minValue):
						v = float(minmax.minValue)
					elif minmax.maxValue and v > float(minmax.maxValue):
						v = float(minmax.maxValue)
					var.set(str(v))
				except ValueError:
					pass

			def on_validate(string):
				regex = re.compile(r"(\+|\-)?[0-9.]*$")
				result = regex.match(string)
				return (string == ""
						or (string.count('+') <= 1
							and string.count('-') <= 1
							and string.count(',') <= 1
							and result is not None
							and result.group(0) != ""))

			var.trace_add("write", callback)
			widget = tk.Entry(master=master, validate="key", textvariable=var)
			vcmd = (widget.register(on_validate), '%P')
			widget.config(validatecommand=vcmd)
		else:
			widget = tk.Entry(master=master)
		widget.insert(0, value)
		if readonly:
			widget["state"] = "disabled"
	elif propType[0] == 's':
		if t[-1].find('~') >= 0:
			options = t[-1].split('~')[1:]
			widget = ttk.Combobox(master=master)
			widget["values"] = options
			widget.set(prop.p["value"])
		else:
			multiline = True if len(t) > 1 and t[1] != '0' else False
			if multiline:
				widget = ScrolledText(master=master, height=1 + multiline * 3)
				widget.insert(tk.INSERT, prop.p["value"])
			else:
				widget = tk.Entry(master=master)
				widget.insert(0, prop.p["value"])
		if readonly:
			widget["state"] = "disabled"
	elif propType[0] == 'p':
		widget = tk.Button(master=master, text=prop.p["name"], command=prop.p["value"])
		callback = PropertyCallback(widget, prop.p["id"], propType)
		widget.anchor('w')

	if widget and propType[0] != 'p':
		callback = PropertyCallback(widget, prop.p["id"], propType)
		widget.anchor('w')

	return widget, callback