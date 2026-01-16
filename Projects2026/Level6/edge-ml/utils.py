import time

def benchmark(func, args):
    start = time.time()
    func(*args)
    return (time.time() - start) * 1000
