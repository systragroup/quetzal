import sys
import os
import json
import boto3
import shutil
from subprocess import Popen, PIPE, STDOUT

sys.path.insert(0, os.path.abspath('quetzal'))

s3 = boto3.resource('s3')


def download_s3_folder(bucket_name, s3_folder, local_dir='/tmp'):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def upload_s3_folder(bucket_name, folder, local_dir='/tmp', metadata={}):
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
            s3_path = os.path.join(folder, os.path.relpath(root, local_dir), file)
            bucket.upload_file(local_path, s3_path, ExtraArgs={'Metadata': metadata})


def clean_folder(folder='/tmp'):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def handler(event, context):
    notebook = event['notebook_path']
    print(event)

    bucket_name = os.environ['BUCKET_NAME']
    # Move (and download) model data and inputs to ephemeral storage
    clean_folder()  # Clean ephemeral storage


    shutil.copytree('./inputs', '/tmp/inputs')
    download_s3_folder(bucket_name, event['scenario_path_S3'])
    arg = json.dumps(event['launcher_arg'])

    file = os.path.join('/tmp', os.path.basename(notebook).replace('.ipynb', '.py'))
    os.system('jupyter nbconvert --to python %s --output %s' % (notebook, file))
    cwd = os.path.dirname(notebook)


    command_list = ['python', file, arg]

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = os.pathsep.join(sys.path)
    process = Popen(command_list, stdout=PIPE, stderr=STDOUT, env=my_env, cwd=cwd)
    process.wait(timeout=500)

    content = process.stdout.read().decode("utf-8")

    if 'Error' in content and "end_of_notebook" not in content:
        print(content)
        raise RuntimeError("Error on execution")

    os.remove(file)
    shutil.rmtree('/tmp/inputs')
    upload_s3_folder(bucket_name, event['scenario_path_S3'], metadata=event.get('metadata', {}))

    return event
