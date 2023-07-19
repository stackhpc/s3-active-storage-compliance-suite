import time
import datetime
import s3fs
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# List of running proxies to benchmark against each other
PROXY_URLS = {
    "active-storage": "http://localhost:8080",
    # 'python': 'http://localhost:8000',
}

# S3_SOURCE = 'http://localhost:9000'
S3_SOURCE = "http://127.0.0.1:9000"
AUTH = ("minioadmin", "minioadmin")
BUCKET = Path("benchmark-data")

N_repeats = 100  # Number of repetitions to average over for each test
# Ns = (1, 1, 1)
Ns = (100, 1000, 2000)  # Dimensions of square chunk sizes to benchmark

s3_fs = s3fs.S3FileSystem(
    key=AUTH[0], secret=AUTH[1], client_kwargs={"endpoint_url": S3_SOURCE}
)

# Make sure the S3 bucket
try:
    s3_fs.mkdir(BUCKET)
except FileExistsError:
    pass


def sum_without_proxy(request_data, filename):
    """Fetches full file from S3 and performs sum locally"""
    dtype = request_data["dtype"]
    with s3_fs.open(BUCKET / filename, "rb") as file:
        arr = np.frombuffer(file.read(), dtype=dtype)
        result = np.sum(arr, dtype=dtype)
        return result


fig, axs = plt.subplots(figsize=(3.5 * len(Ns), 6), ncols=len(Ns))
for i, array_dim in enumerate(Ns):
    # Create test array
    filename = "test-data.dat"
    X = np.random.rand(array_dim, array_dim).astype("float32")
    N_bytes = X.itemsize * X.size / 1024**2
    print(
        f"Starting benchmark for {array_dim} x {array_dim} array (file size = {N_bytes:.2f} MB)"
    )

    # Upload to S3
    print("Uploading test file to S3")
    with s3_fs.open(BUCKET / filename, "wb") as file:
        file.write(X.tobytes())

    # Run benchmark
    request_data = {
        "source": S3_SOURCE,
        "bucket": str(BUCKET),
        "object": filename,
        "dtype": str(X.dtype),
    }

    timings = {}

    # Check timings without proxy for a reference timescale
    times = []
    print("Starting baseline benchmark without proxy")
    for n in tqdm(range(N_repeats)):
        t_start = time.perf_counter()
        sum_without_proxy(request_data, filename)
        t_end = time.perf_counter()
        times.append(t_end - t_start)
    timings["no-active-storage"] = times

    # Make use of proxy
    for name, url in PROXY_URLS.items():
        times = []
        print("Starting benchmarks for proxy running on", url)
        for n in tqdm(range(N_repeats)):
            t_start = time.perf_counter()
            response = requests.post(url + "/v1/sum/", json=request_data, auth=AUTH)
            if response.status_code != 200:
                raise Exception("Proxy request failed. Error message: " + response.text)
            t_end = time.perf_counter()
            times.append(t_end - t_start)
        timings[name] = times

    # Create figure to display results
    axs[i].boxplot(timings.values(), vert=True, labels=timings.keys(), whis=(0, 100))
    axs[i].set_title(f"Chunk size = {N_bytes:.2f} MB")
    axs[i].set_ylabel("Average time to perform reduction (seconds)")
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=0)
    axs[i].set_ylim((0, axs[i].get_ylim()[1]))
    print()

# Save figure to disk
fig.suptitle("Python S3 active storage proxy benchmarks")
timestamp = datetime.datetime.now().strftime("%Y-%d-%m--%H:%M:%S")
output_file = f"benchmark--{timestamp}.png"
plt.tight_layout()
plt.savefig(output_file)

# Remove benchmark data from S3
s3_fs.rm(str(BUCKET), recursive=True)
