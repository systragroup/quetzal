import os
import sys
import boto3
import logging


s3 = boto3.resource('s3')


def upload_s3_folder(bucket_name, folder, local_dir):
    """
    Upload the contents of a folder directory to S3
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(folder, root, file)
            bucket.upload_file(local_path, s3_path)


def main():
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

    paths = ['inputs/pt/links.geojson',
             'inputs/pt/nodes.geojson',
             'inputs/road/road_links.geojson',
             'inputs/road/road_nodes.geojson',
             'outputs/',
             'inputs/params.json']

    bucket = s3.Bucket(os.environ["AWS_BUCKET_NAME"])

    for scenario in sys.argv[2:]:
        # Delete content
        for obj in bucket.objects.filter(Prefix=scenario):
            s3.Object(bucket.name, obj.key).delete()

        print(f"Updating {scenario} scenario")
        for path in paths:
            if (path == '') | (path is None):
                continue
            if not os.path.exists(path):
                print(f"Local path does not exists: {path}")
                continue
            if os.path.isdir(path):
                print(f"Uploading {path} folder")
                upload_s3_folder(os.environ["AWS_BUCKET_NAME"], scenario, path)
            else:
                print(f"Uploading {path}")
                bucket.upload_file(path, os.path.join(scenario, path))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: At least two argument is required.")
        print("Usage: python {name} model_folder scenario1 [scenario2] ...".format(name=sys.argv[0]))
        sys.exit(1)

    source = os.path.dirname(os.path.abspath(__file__))
    quetzal_root = os.path.abspath(os.path.join(source, '../../..'))
    os.chdir(os.path.abspath(os.path.join(quetzal_root, sys.argv[1])))
    main()
