
import os
from botocore.exceptions import ClientError
import numpy as np
from compliance.config import BUCKET_NAME

def upload_to_s3(s3_client, arr: np.array, filename: str) -> None:

    """ Upload a numpy array in binary format to an S3 storage bucket """

    #Create required bucket if it doesn't yet exist
    try:
        bucket = s3_client.create_bucket(Bucket=BUCKET_NAME)
    except ClientError:
        pass #Bucket already exists

    #Write file to disk, upload it to S3, then remove on-disk file
    try:
        arr.tofile(filename)
        s3_client.upload_file(filename, BUCKET_NAME, filename.replace('.npy', '.bin'))
    finally:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    return

def fetch_from_s3(s3_client, filename: str) -> bytes:
    """ Fetches data from configured S3 source and returns the content as raw bytes """
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
    content = response['Body'].read()
    return content