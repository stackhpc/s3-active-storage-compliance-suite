
import time
import datetime
import s3fs
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

#List of running proxies to benchmark against each other
PROXY_URLS = {
    'active-storage': 'http://localhost:8000',
}

# Configure upstream object store details
S3_SOURCE = 'http://localhost:9000'
AUTH = ('minioadmin', 'minioadmin')
BUCKET = Path('benchmark-data')

# Dimensions of square chunk sizes to benchmark
SQUARE_DIMS = range(100, 1100, 300)

def sum_without_proxy(fs, filename, request_data, expected_ans):
    """Fetches full file from S3 and performs sum locally"""
    with fs.open(BUCKET / filename, "rb") as file:
        arr = np.frombuffer(file.read(), dtype=request_data['dtype'])
        result = np.sum(arr, dtype=request_data['dtype'])
        if not np.isclose(result, expected_ans):
            raise Exception(f'No-proxy case returned incorrect answer: {result} (expected {expected_ans})')
    return


def sum_with_proxy(proxy_url, request_data, expected_ans):
    """Requests sum result from active storage proxy"""
    response = requests.post(proxy_url + '/v1/sum/', json=request_data, auth=AUTH)
    if response.status_code != 200:
        raise Exception("Proxy request failed. Error message: " + response.text)
    result = np.frombuffer(response.content, dtype=request_data['dtype'])[0]
    if not np.isclose(result, expected_ans):
        raise Exception(f'No-proxy case returned incorrect answer: {result} (expected {expected_ans})')
    return

def repeat(x, n):
    for _ in range(n):
        yield x

def main():

    timings = {}

    # Initialize S3 interation stuff
    s3_fs = s3fs.S3FileSystem(key=AUTH[0], secret=AUTH[1], client_kwargs={'endpoint_url': S3_SOURCE})

    # Make sure the S3 bucket
    try:
        s3_fs.mkdir(BUCKET)
    except FileExistsError:
        pass

    for N in SQUARE_DIMS:

        print(f'Starting {N}x{N} array benchmarks')
        run_timings = {}

        # Generate and upload the test data
        filename = f'test-data.dat'
        X = np.random.rand(N, N).astype('float32')
        N_bytes = X.itemsize * X.size / 1024**2
        with s3_fs.open(BUCKET / filename, 'wb') as file:
            file.write(X.tobytes())

        run_timings['chunk-size-MB'] = N_bytes

        request_data = {
            'source': S3_SOURCE,
            'bucket': str(BUCKET),
            'object': filename,
            'dtype': str(X.dtype),
        }

        # Run no-proxy benchmark
        print('-> No proxy:')
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            _ = list(tqdm(executor.map(sum_without_proxy, repeat(s3_fs, REQUESTS), repeat(filename, REQUESTS), repeat(request_data, REQUESTS), repeat(X.sum(), REQUESTS)), total=REQUESTS))
        run_timings['no-proxy'] = time.perf_counter() - t_start

        for name, url in PROXY_URLS.items():
            print(f'-> {name}:')
            t_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                _ = list(tqdm(executor.map(sum_with_proxy, repeat(url, REQUESTS), repeat(request_data, REQUESTS), repeat(X.sum(), REQUESTS)), total=REQUESTS))
            run_timings[name] = time.perf_counter() - t_start

        timings[N] = run_timings

    return timings

if __name__ == '__main__':
    
    # Make plot
    WORKERS_AND_REQUEST = [(2, 10), (4, 10), (8, 10)]
    fig, axs = plt.subplots(figsize=(8, 4*len(WORKERS_AND_REQUEST)), nrows=len(WORKERS_AND_REQUEST))
    fig.suptitle("S3 active storage proxy benchmarks (parallel requests)")

    for ax, (WORKERS, REQUESTS) in zip(axs, WORKERS_AND_REQUEST):    

        print(f'\n Starting benchmark run with {WORKERS} and {REQUESTS} requests')
        print('---------------------------------------------------n')
        results = main()

        for type in ['no-proxy', *PROXY_URLS]:
            xdata = [results[N]['chunk-size-MB'] for N in SQUARE_DIMS]
            ydata = [results[N][type] for N in SQUARE_DIMS]
            ax.scatter(xdata, ydata, label=type)
            ax.plot(xdata, ydata)

        ax.set_xlabel("Chunk size (MB)")
        ax.set_ylabel(f"Time taken to handle {REQUESTS} total requests\nfrom {WORKERS} parallel workers (seconds)")
        ax.legend()
        ax.set_title(f'Parallel workers: {WORKERS} & Total requests: {REQUESTS}')

    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    output_file = f'benchmark-parallel--{timestamp}.png'
    plt.tight_layout()
    plt.savefig(output_file)
