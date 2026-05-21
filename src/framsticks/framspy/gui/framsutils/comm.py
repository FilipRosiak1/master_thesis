import asyncore
import socket
from io import StringIO
from typing import Callable, Dict
import threading
import asyncio
import queue
import time

class EventConsumer(threading.Thread):
	def __init__(self) -> None:
		threading.Thread.__init__(self)
		self.condition = threading.Condition()
		self.queue = queue.Queue()
		self.is_running = True
		self.runningChangeEventCallback = None
		self.populationsGroupChangeEventCallback = None
		self.genepoolsGroupChangeEventCallback = None
		self.messagesEventCallback = None

	def run(self):
		while self.is_running:
			if self.queue.empty():
				time.sleep(0.0001)
			else:
				block = self.queue.get()
				#print("------- EVENT -------")
				#print(block)
				#print("------- END EVENT -------")
				header: str = block[:block.find('\n')]
				if header.find("running_changed") != -1 and self.runningChangeEventCallback:
					self.runningChangeEventCallback(block, header)
				elif header.find("populations/groups_changed") != -1 and self.populationsGroupChangeEventCallback:
					self.populationsGroupChangeEventCallback(block, header)
				elif header.find("genepools/groups_changed") != -1 and self.genepoolsGroupChangeEventCallback:
					self.genepoolsGroupChangeEventCallback(block, header)
				elif header.find("messages") != -1 and self.messagesEventCallback:
					self.messagesEventCallback(block, header)

	def stop(self):
		self.is_running = False

class CommWrapper(object):
	def start(self, host: str, port: int):
		self.client = Comm(host, port, self.onClose)
		self.thread = threading.Thread(target=asyncore.loop, kwargs={'timeout': 0.0005})
		self.thread.start()
		self.client.write_data("version 5")
		self.client.write_data("use request_id")
		self.client.write_data("use needfile_id")
		self.client.write_data("use groupnames")
		self.connected = True

	def stop(self):
		self.client.close()
		self.thread.join()

	def onClose(self):
		self.connected = False

	def close(self):
		self.thread.join()

	async def read(self, i: int, timeout: float) -> str:
		data = ""
		try:
			data = await asyncio.wait_for(self._read(i), timeout=timeout)
		except:
			pass
		#print("response")
		#print(repr(data))
		return data

	async def _read(self, i: int) -> str:
		while True:
			if i in self.client.blocks:
				return self.client.blocks.pop(i)
			try:
				await asyncio.sleep(0.005)
			except asyncio.CancelledError:
				break

	def write(self, data: str, index = -1) -> int:
		command = data.split(' ')[0]
		if command in ["info", "set", "get", "call", "reg", "file"]:
			if index == -1:
				index = self.client.getIndex()
			data = command + ' r' + str(index) + data[len(command):]
		self.client.write_data(data)
		return index

class Comm(asyncore.dispatcher):
	def __init__(self, host: str, port: int, onClose: Callable[[None], None] = None) -> None:
		self.onClose = onClose

		asyncore.dispatcher.__init__(self)
		self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
		self.host = host
		self.port = port
		
		self.connect((self.host, self.port))
		self.out_buffer = "".encode()
		self.in_buffer = StringIO()
		self.blocks: Dict[str] = {}
		self._tmpbuff = StringIO()
		self.index = 1
		self.is_embedded = False

		self.consumer = EventConsumer()
		self.consumer.start()
	
	def writable(self):
		is_writable = (len(self.out_buffer) > 0)
		return is_writable

	def readable(self):
		return True

	def handle_close(self):
		self.close()

	def close(self) -> None:
		self.in_buffer.close()
		self._tmpbuff.close()
		self.consumer.stop()
		if self.onClose:
			self.onClose()
		super().close()

	def handle_connect(self):
		pass

	def handle_read(self):
		rec_msg = self.recv(1024)
		self.in_buffer.write(rec_msg.decode())
		self.parse()

	def handle_write(self) -> None:
		sent = self.send(self.out_buffer)
		self.out_buffer = self.out_buffer[sent:]

	def write_data(self, data: str):
		#print("request")
		#print(repr(data))
		self.out_buffer += (data + '\n' if data[-1] != '\n' else '').encode()

	def read_data(self):
		return self.in_buffer.getvalue()

	def parse(self):
		self.in_buffer.seek(0)
		for line in self.in_buffer:
			if line.endswith('\n'):
				self._tmpbuff.write(line)
				if line.endswith(":~\n"):
					self.is_embedded = True
				if self.is_embedded:
					if line.endswith("~\n") and not line.endswith("\~\n"):
						self.is_embedded = False
				else:
					if line.startswith("eof"):
						block = self._tmpbuff.getvalue().strip()
						if block.startswith("event"):
							self.consumer.queue.put(block)
							self._tmpbuff.truncate(0)
							self._tmpbuff.seek(0)
					elif line.startswith("ok") or line.startswith("error") or line.startswith("needfile"):
						block = self._tmpbuff.getvalue()
						if line.find(' ') > -1:
							self.blocks[int(line.split(' ')[1][1:])] = block
						self._tmpbuff.truncate(0)
						self._tmpbuff.seek(0)
		self.in_buffer.truncate(0)
		self.in_buffer.seek(0)
		if not line.endswith('\n'):
			self.in_buffer.write(line)

	def getIndex(self) -> int:
		# Not every response could be fetched, because of timeout of request.
		# Stale responses needs to be cleaned to prevent stack growing.
		r = 100	# Remove all messagess older than r
		if self.index % r != (self.index + 1) % r:
			self.blocks = {k: v for k, v in self.blocks.items() if k > self.index - r}
		self.index += 1
		return self.index - 1
