from thread_module import myThread

def test_function(file_id):
	print file_id

def test_function_1(file_id, number):
	print file_id, number

def test_function_2(file_id, number1, number2):
	print file_id, number1, number2

thread0 = myThread(0, 0, 0, test_function)
thread1 = myThread(1, 10, 10, test_function_1, 10)
thread2 = myThread(2, 20, 20, test_function_2, 10, 20)

thread0.start()
thread1.start()
thread2.start()