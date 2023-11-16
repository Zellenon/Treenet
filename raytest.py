import ray
import time

ray.init()

@ray.remote
def f(x):
    time.sleep(0.1)
    return x*x

def f2(x):
    time.sleep(0.1)
    return x*x

def parmap(f, list):
    return [f.remote(x) for x in list]

# Call parmap() on a list consisting of first 5 integers.
start_time = time.perf_counter()
result_ids = parmap(f, range(1, 600))
results = ray.get(result_ids)
end_time = time.perf_counter()
print(f"The execution time is: {end_time - start_time}")


start_time = time.perf_counter()
results = [f2(w) for w in range(1,600)]
end_time = time.perf_counter()
print(f"The execution time is: {end_time - start_time}")



