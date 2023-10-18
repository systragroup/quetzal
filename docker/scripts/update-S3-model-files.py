import os
import sys
import boto3
import logging

# python update-S3-model-files.py quetzal_test base
# copy files from quetzal_test/scenarios/base/
# to base/ on s3.

s3 = boto3.resource('s3')

def list_paths_in_directory(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    return file_paths

def main():
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

    bucket = s3.Bucket(os.environ["AWS_BUCKET_NAME"])
    for scenario in sys.argv[2:]:
        # Delete content
        for obj in bucket.objects.filter(Prefix=scenario):
            s3.Object(bucket.name, obj.key).delete()

        print(f"Updating {scenario} scenario")
        localpath = 'scenarios/' + scenario + '/' 
        if not os.path.exists(localpath):
            print(f"Local path does not exists: {localpath}")
            continue
        if os.path.isdir(localpath):
            files = list_paths_in_directory(localpath)
            for file in files:
                print('upload:',file[10:])
                bucket.upload_file(file, file[10:])


          


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: At least two argument is required.")
        print("Usage: python {name} model_folder scenario1 [scenario2] ...".format(name=sys.argv[0]))
        sys.exit(1)

    source = os.path.dirname(os.path.abspath(__file__))
    quetzal_root = os.path.abspath(os.path.join(source, '../../..'))
    os.chdir(os.path.abspath(os.path.join(quetzal_root, sys.argv[1])))
    main()
