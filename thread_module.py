import threading

class myThread(threading.Thread):
	def __init__(self, thread_id, start_id, finish_id, function):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.start_id = start_id
		self.finish_id = finish_id
		self.function = function

	def run(self):
		print "Starting thread", self.thread_id
		for file_id in range(self.start_id, self.finish_id + 1):
			self.function(file_id)
		print "Exiting thread", self.thread_id 