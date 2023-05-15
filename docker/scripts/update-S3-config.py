import os
import sys
import boto3


s3 = boto3.resource('s3')


def main():
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

    bucket = s3.Bucket(os.environ["AWS_BUCKET_NAME"])
    bucket.upload_file('quenedi.config.json', "quenedi.config.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: At least one argument is required.")
        print("Usage: python {name} model_folder".format(name=sys.argv[0]))
        sys.exit(1)

    source = os.path.dirname(os.path.abspath(__file__))
    quetzal_root = os.path.abspath(os.path.join(source, '../../..'))
    os.chdir(os.path.abspath(os.path.join(quetzal_root, sys.argv[1])))
    main()
