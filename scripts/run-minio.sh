# Use anon storage volume so that test data is removed when container is stopped
exec docker run --rm -p 9000:9000 -p 9001:9001 -v :/data minio/minio server data --console-address ":9001"