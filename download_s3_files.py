import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def download_files_from_s3():
    """
    Downloads specified files from the S3 bucket to the local directory.
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    # Define the bucket name
    bucket_name = 'breach-predictor-data-anuda'

    # List of files to download
    files = [
        ('breach_predictor_model.pkl', 'breach_predictor_model.pkl'),
        ('breach_data.csv', 'breach_data.csv'),
        ('company_profiles.csv', 'company_profiles.csv')
    ]

    # Download each file with error handling
    for key, local_filename in files:
        try:
            s3.download_file(bucket_name, key, local_filename)
            print(f"Downloaded {key} successfully!")
        except NoCredentialsError:
            print(f"Credentials not available for downloading {key}.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"The object {key} does not exist in the bucket.")
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                print(f"The bucket {bucket_name} does not exist.")
            else:
                print(f"An error occurred while downloading {key}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while downloading {key}: {e}")

    print("Download process completed!")

if __name__ == "__main__":
    download_files_from_s3()
