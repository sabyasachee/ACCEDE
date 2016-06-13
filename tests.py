from thread_module import myThread
from time import sleep
from mpi4py import MPI
from multiprocessing import Process, Queue

def test_function(file_id):
	print file_id

def test_function_1(file_id, number):
	print file_id, number

def test_function_2(file_id, number1, number2):
	print file_id, number1, number2

def test_screen():
	while True:
		print "Hello"
		sleep(5)
		pass

values = []
ps = []

def func(i, value, q):
	q.put((i, value))

q = Queue()
for i in range(0, 5):
	values.append(0)
	p = Process(target = func, args = (i, 5*i + 1, q,))
	ps.append(p)
	p.start()

for p in ps:
	p.join()

while not q.empty():
	i, value = q.get()
	values[i] = value

print values
# for (i, value) in q:
# 	print i, value
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# func(rank, rank*5)
# print values