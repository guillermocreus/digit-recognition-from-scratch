import threading
import queue

queue = queue.Queue()

def MyThread1():
    A1, A2, l1, l2, v1, v2 = train_test_data()
    queue.put([A1, A2, l1, l2, v1, v2])
def MyThread2():
    A, B, c, d, e, f = train_test_data()
    queue.put([A, B, c, d, e, f])

t1 = threading.Thread(target=MyThread1, args=[])
t2 = threading.Thread(target=MyThread2, args=[])

t1.start()
t2.start()
t1.join()
t2.join()

result1 = queue.get()
result2 = queue.get()