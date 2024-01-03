import sys
import os
import json
import boto3
import shutil
import time
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


def format_error(err):
    # return the error starting a Traceback.
    part = err.partition('Traceback')
    res = part[1] + part[2]
    if len(res) > 0:
        return res
    else:
        return part[0]


def handler(event, context):
    t0 = time.time()
    notebook = event['notebook_path']
    print(event)

    bucket_name = os.environ['BUCKET_NAME']
    # Move (and download) model data and inputs to ephemeral storage
    clean_folder()  # Clean ephemeral storage

    try:
        shutil.copytree('./inputs', '/tmp/inputs')
    except:
        print('cannot copy local docker inputs/ folder. its maybe missing on purpose')
   
    download_s3_folder(bucket_name, event['scenario_path_S3'])
    t1 = time.time()
    print('Download inputs from s3 and docker image: {} seconds'.format(t1 - t0))

    arg = json.dumps(event['launcher_arg'])
    print(arg)

    file = os.path.join('/tmp', os.path.basename(notebook).replace('.ipynb', '.py'))
    if notebook.endswith('.ipynb'):
        os.system('jupyter nbconvert --to python %s --output %s' % (notebook, file))
    else:
        os.system('cp %s %s' % (notebook, file))
    cwd = os.path.dirname(notebook)
    if cwd =='':
        cwd = '/'
    command_list = ['python', file, arg]
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = os.pathsep.join(sys.path)

    t2=time.time()
    print('Notebook conversion: {} seconds'.format(t2 - t1))

    process = Popen(command_list, stdout=PIPE, stderr=STDOUT, env=my_env, cwd=cwd)
    process.wait(timeout=800)

    content = process.stdout.read().decode("utf-8")
    t3 = time.time()
    print('Notebook execution: {} seconds'.format(t3 - t2))

    print(content)
    if 'Error' in content and "end_of_notebook" not in content:
        raise RuntimeError(format_error(content))

    os.remove(file)
    shutil.rmtree('/tmp/inputs')
    upload_s3_folder(bucket_name, event['scenario_path_S3'], metadata=event.get('metadata', {}))
    t4 = time.time()
    print('Upload to S3: {} seconds'.format(t4 - t3))
    print('Total excecution time: {} seconds'.format(t4 - t0))

    return event
