import threading

class myThread(threading.Thread):
	def __init__(self, thread_id, start_id, finish_id, function, *args):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.start_id = start_id
		self.finish_id = finish_id
		self.function = function
		self.args = args

	def run(self):
		print "Starting thread", self.thread_id
		for file_id in range(self.start_id, self.finish_id + 1):
			self.function(file_id, *self.args)
		print "Exiting thread", self.thread_id 

def process_multithread(function, *args):
	thread0 = myThread(0, 0, 999, function, *args)
	thread1 = myThread(1, 1000, 1999, function, *args)
	thread2 = myThread(2, 2000, 2999, function, *args)
	thread3 = myThread(3, 3000, 3999, function, *args)
	thread4 = myThread(4, 4000, 4999, function, *args)
	thread5 = myThread(5, 5000, 5999, function, *args)
	thread6 = myThread(6, 6000, 6999, function, *args)
	thread7 = myThread(7, 7000, 7999, function, *args)
	thread8 = myThread(8, 8000, 8999, function, *args)
	thread9 = myThread(9, 9000, 9799, function, *args)

	thread0.start()
	thread1.start()
	thread2.start()
	thread3.start()
	thread4.start()
	thread5.start()
	thread6.start()
	thread7.start()
	thread8.start()
	thread9.start()