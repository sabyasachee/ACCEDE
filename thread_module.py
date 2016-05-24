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

def process_multithread(function):
	thread0 = myThread(0, 0, 999, function)
	thread1 = myThread(1, 1000, 1999, function)
	thread2 = myThread(2, 2000, 2999, function)
	thread3 = myThread(3, 3000, 3999, function)
	thread4 = myThread(4, 4000, 4999, function)
	thread5 = myThread(5, 5000, 5999, function)
	thread6 = myThread(6, 6000, 6999, function)
	thread7 = myThread(7, 7000, 7999, function)
	thread8 = myThread(8, 8000, 8999, function)
	thread9 = myThread(9, 9000, 9799, function)

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