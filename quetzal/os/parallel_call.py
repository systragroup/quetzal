import json
import os
import sys
import shutil
import time
import uuid
from subprocess import Popen


def add_freeze_support(file):
    with open(file, 'r') as pyfile:
        lines = pyfile.readlines()
    top = [
        "def main():\n"
    ]
    bottom = [
        "from multiprocessing import freeze_support\n"
        "if __name__ == '__main__':\n"
        "    freeze_support()\n"
        "    main()\n"
    ]
    lines = top + ['    ' + line for line in lines] + bottom
    with open(file, 'w') as pyfile:
        pyfile.writelines(lines)

def split_cwd_and_file(path):
    path_list = path.split('/')
    if len(path_list) != 1:
        relpath = '/'.join(path_list[:-1])
        cwd = os.path.abspath(relpath)
    else:
        cwd = os.getcwd()
    return cwd, path_list[-1]


def parallel_call_jobs(jobs, mode='w', leave=False, workers=1, sleep=1):
    popens = {}
    for i, file, arg, stdout_file, stderr_file in jobs:
        cwd, file = split_cwd_and_file(file)
        print(i, file, arg)
        with open(stdout_file, mode) as stdout:
            with open(stderr_file, mode) as stderr:
                if type(arg) == tuple:
                    command_list = ['python', file] +list(arg)
                else:
                    command_list = ['python', file] +[arg]
                my_env = os.environ
                my_env["PYTHONPATH"] = os.pathsep.join(sys.path)[1:]
                popens[i] = Popen(
                    command_list,
                    stdout=stdout,
                    stderr=stderr,
                    env=my_env,
                    cwd=cwd
                )
                if (i + 1) % workers == 0 or (i + 1) == len(jobs):
                    # print('waiting')
                    for p in popens.values():
                        p.wait()
        time.sleep(sleep)

    for i, file, arg, stdout_file, stderr_file in jobs:
        cwd, file = split_cwd_and_file(file)
        subprocess_name = str(file) + str(i)
        if not leave:
            try:
                os.remove(os.path.join(cwd,file))
            except FileNotFoundError:
                pass
        with open(stderr_file, 'r', encoding='latin') as stderr:
            content = stderr.read()
            if 'Error' in content and "end_of_notebook" not in content:
                print("subprocess **{} {}** terminated with an error.".format(subprocess_name, arg))


def parallel_call_notebooks(
    *notebook_arg_list_tuples,
    stdout_path='out.txt',
    stderr_path='err.txt',
    workers=2,
    sleep=1,
    leave=False,
    errout_suffix=False,
    freeze_support=True,
    return_jobs=False
):
    start = time.time()
    mode = 'w' if errout_suffix else 'a+'
    jobs = []
    outer_i = 0
    for notebook, arg_list in notebook_arg_list_tuples:
        os.system('jupyter nbconvert --to python %s' % notebook)
        file = notebook.replace('.ipynb', '.py')
        if freeze_support:
            add_freeze_support(file)

        for i in range(len(arg_list)):
            arg = arg_list[i]
            suffix = ''.join(arg) if errout_suffix else ''
            suffix += '_' + file.split('/')[-1].split('.')[0]
            stdout_file = stdout_path.replace('.txt', '_' + suffix + '.txt')
            stderr_file = stderr_path.replace('.txt', '_' + suffix + '.txt')
            jobs.append([outer_i, file, arg, stdout_file, stderr_file])
            outer_i += 1

    parallel_call_jobs(jobs, mode=mode, leave=leave, workers=workers, sleep=sleep)
    end = time.time()
    print(int(end - start), 'seconds')
    if return_jobs: return jobs


def parallel_call_notebook(
    notebook,
    arg_list,
    stdout_path='out.txt',
    stderr_path='err.txt',
    workers=2,
    sleep=1,
    leave=False,
    errout_suffix=False,
    freeze_support=True,
    return_jobs=False
):

    start = time.time()
    jobs = []
    os.system('jupyter nbconvert --to python %s' % notebook)
    file = notebook.replace('.ipynb', '.py')
    if freeze_support:
        add_freeze_support(file)

    mode = 'w' if errout_suffix else 'a+'

    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = ''.join(arg) if errout_suffix else ''
        suffix += '_' + file.split('/')[-1].split('.')[0]
        stdout_file = stdout_path.replace('.txt', '_' + suffix + '.txt')
        stderr_file = stderr_path.replace('.txt', '_' + suffix + '.txt')
        jobs.append([i, file, arg, stdout_file, stderr_file])

    parallel_call_jobs(jobs, mode=mode, leave=leave, workers=workers, sleep=sleep)
    end = time.time()
    print(int(end - start), 'seconds')
    if return_jobs: return jobs


def parallel_call_python(
    file,
    arg_list,
    stdout_path='out.txt',
    stderr_path='err.txt',
    workers=2,
    sleep=0,
    errout_suffix=False,
    process_name='process',
):
    popens = {}
    notebook = file

    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = arg if errout_suffix else ''
        suffix += '_' + process_name
        mode = 'w' if errout_suffix else 'a+'
        # print(i, arg)
        with open(stdout_path.replace('.txt', '_' + suffix + '.txt'), mode) as stdout:
            with open(stderr_path.replace('.txt', '_' + suffix + '.txt'), mode) as stderr:
                popens[i] = Popen(
                    ['python', file, arg],
                    stdout=stdout,
                    stderr=stderr
                )
                if (i + 1) % workers == 0 or (i + 1) == len(arg_list):
                    # print('waiting')
                    for p in popens.values():
                        p.wait()
        time.sleep(sleep)

    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = arg if errout_suffix else ''
        suffix += '_' + process_name
        mode = 'r'
        with open(stderr_path.replace('.txt', '_' + suffix + '.txt'), mode) as stderr:
            content = stderr.read()
            if 'Error' in content and "end_of_notebook" not in content:
                print("subprocess **{} {}** terminated with an error.".format(i, arg))
    return popens


def parallel_call_subprocess(
    subprocess_filepath,
    kwarg_list,
    dump_kwargs=json.dump,
    load_result=json.load,
    leave=False,
    *args,
    **kwargs
):
    input_files = []
    output_files = []
    uids = []
    io_strings = []

    if not os.path.exists('subprocess_io/'):
        os.mkdir('subprocess_io')

    for kwarg_dict in kwarg_list:
        uid = str(uuid.uuid4())

        input_file = r'subprocess_io/input' + '-' + uid + '.json'
        output_file = r'subprocess_io/output' + '-' + uid + '.json'
        output_files.append(output_file)

        dump_kwargs(kwarg_dict, input_file)
        io_string = json.dumps({'input_json': input_file, 'output_json': output_file})
        io_strings.append(io_string)

    parallel_call_python(
        subprocess_filepath,
        io_strings,
        errout_suffix=False,
        stdout_path='subprocess_io/out.txt',
        stderr_path='subprocess_io/err.txt',
        *args,
        **kwargs,
    )
    results = [load_result(file) for file in output_files]

    if not leave:
        shutil.rmtree('subprocess_io')
    return results
