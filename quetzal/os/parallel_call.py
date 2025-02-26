import json
import os
import sys
import shutil
import time
import uuid
from subprocess import Popen
import string

import nbformat
from nbconvert import PythonExporter


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
                #my_env = os.environ
                #my_env["PYTHONPATH"] = os.pathsep.join(sys.path)[1:]
                popens[i] = Popen(
                    command_list,
                    stdout=stdout,
                    stderr=stderr,
                    #env=my_env,
                    cwd=cwd
                )
                if (i + 1) % workers == 0 or (i + 1) == len(jobs):
                    # print('waiting')
                    for p in popens.values():
                        p.wait()
                        p.terminate()
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
    notebook_list,
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
    files = []
    for notebook in notebook_list:
        file = notebook.replace('.ipynb', '.py')
        files.append(file)
        if not os.path.exists(file):
            os.system('jupyter nbconvert --to python %s' % notebook)
        if not os.path.exists(file): # jupyter nb convert failed, for instance, jupyter is not recognized
            convertNotebook(notebook, file)
            if freeze_support:
                add_freeze_support(file)
        

    mode = 'w' if errout_suffix else 'a+'

    supported_characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '-_' 
    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = ''
        if errout_suffix:
            try:
                temp = json.loads(arg)
                suffix = '_'.join(temp.values())
            except (json.JSONDecodeError, TypeError):
                suffix = str(arg)
        suffix += '_' + files[i].split('/')[-1].split('.')[0]
        suffix = ''.join([s for s in suffix if s in supported_characters])
        stdout_file = stdout_path.replace('.txt', '_' + suffix + '.txt')
        stderr_file = stderr_path.replace('.txt', '_' + suffix + '.txt')
        jobs.append([i, files[i], arg, stdout_file, stderr_file])

    parallel_call_jobs(jobs, mode=mode, leave=leave, workers=workers, sleep=sleep)
    end = time.time()
    print(int(end - start), 'seconds')
    if return_jobs: return jobs



def convertNotebook(notebookPath, modulePath):

  with open(notebookPath) as fh:
    nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

  exporter = PythonExporter()
  source, meta = exporter.from_notebook_node(nb)

  with open(modulePath, 'w+') as fh:
    fh.writelines(source)

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

    supported_characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + '-_' 
    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = ''
        if errout_suffix:
            try:
                temp = json.loads(arg)
                suffix = '_'.join(temp.values())
            except (json.JSONDecodeError, TypeError):
                suffix = str(arg)
        suffix += '_' + file.split('/')[-1].split('.')[0]
        suffix = ''.join([s for s in suffix if s in supported_characters])
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
                        p.terminate()
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



from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, Manager, Pipe
import sys


def parallel_executor(function, method='pool', num_workers:int=1, parallel_kwargs:dict={}, **kwargs):
    '''
    function: 
        function to paralellize
    method = 'pool' | 'queue' | 'manager' | 'pipe':
        Pool, queue and pipe mostly the same performance.
        Only Pipe works on AWS lambda. so if on Lambda, force method = pipe
    num_workers:
        number of threads.
    parallel_kwargs: dict of arrays
        kwargs in the function to be parallelize. This should be a List.
        ex: dijkstra(indices=parallel_kwargs['indices'][i]) where i is in range(num_workers)
    '''
    on_lambda = bool(os.environ.get('AWS_EXECUTION_ENV'))

    if on_lambda:
        method = 'pipe'

    if method == 'pool':
        results = process_pool_executor(function=function, 
                                        num_workers=num_workers, 
                                        parallel_kwargs=parallel_kwargs, 
                                        **kwargs)
        
    elif method == 'pipe':
         results = process_pipe_executor(function=function, 
                                        num_workers=num_workers, 
                                        parallel_kwargs=parallel_kwargs, 
                                        **kwargs)
                        
    elif method =='queue':
        results = process_queue_executor(function=function,
                                        num_workers=num_workers, 
                                        parallel_kwargs=parallel_kwargs, 
                                        **kwargs)

    elif method == 'manager':
        results = process_manager_executor(function=function, 
                                            num_workers=num_workers, 
                                            parallel_kwargs=parallel_kwargs, 
                                            **kwargs)
        
    else:
        print(method,'is not a valid choice')        
        return 0
        
    return results


def process_pool_executor(function, num_workers=1, parallel_kwargs={}, **kwargs):
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            batch_kwargs = {key:item[i] for key,item in parallel_kwargs.items()} 
            p = executor.submit(
                function,
                **batch_kwargs,
                **kwargs
            )
            results.append(p)
    results  = [res.result() for res in results]
    return results

def process_wrapper(function, process_kwargs, kwargs, queue, i):
    result = function(**process_kwargs,**kwargs)
    queue.put((i,result))


def process_queue_executor(function, num_workers=1, parallel_kwargs={}, **kwargs):
    # as the order in wich the Queue results is extract is not in the order it was created.
    # we keep track of the order and assign the result in the correct index of results
    queue = Queue()
    processes = []
    results = [None] * num_workers
    for i in range(num_workers):
        process_kwargs = {key:item[i] for key,item in parallel_kwargs.items()} 
        process = Process(target=process_wrapper, args=(function, process_kwargs, kwargs, queue, i))
        process.start()
        processes.append(process)
    for _ in processes:
        res = queue.get()
        results[res[0]]=res[1] 
    for process in processes:
        process.join()      


    return results

def process_wrapper_manager(function, process_kwargs, kwargs, result_list, i):
    result = function(**process_kwargs,**kwargs)
    result_list[i] = result


def process_manager_executor(function, num_workers=1, parallel_kwargs={}, **kwargs):
    # as the order in wich the Queue results is extract is not in the order it was created.
    # we keep track of the order and assign the result in the correct index of results
    processes = []
    manager = Manager()
    results = manager.list([None] * num_workers)
    for i in range(num_workers):
        process_kwargs = {key:item[i] for key,item in parallel_kwargs.items()} 
        process = Process(target=process_wrapper_manager, args=(function, process_kwargs, kwargs, results, i))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    # Convert the manager list to a regular list for easier access
    results = np.array(results, dtype="object")   
    return results


def process_wrapper_pipe(function, process_kwargs, kwargs, conn,i):
    result = function(**process_kwargs,**kwargs)
    conn.send((i,result))


def process_pipe_executor(function, num_workers=1, parallel_kwargs={}, **kwargs):
    # This one works on Lambda.
    # as the order in wich the Queue results is extract is not in the order it was created.
    # we keep track of the order and assign the result in the correct index of results
    # https://aws.amazon.com/blogs/compute/parallel-processing-in-python-with-aws-lambda/
    processes = []
    parent_connections = []
    child_connections = []
    results = [None] * num_workers
    for i in range(num_workers):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        child_connections.append(child_conn)
        process_kwargs = {key:item[i] for key,item in parallel_kwargs.items()} 
        process = Process(target=process_wrapper_pipe, args=(function, process_kwargs, kwargs, child_conn,i))
        process.start()
        processes.append(process)
    for i in range(num_workers):
        res = parent_connections[i].recv()
        results[res[0]]=res[1] 
        child_connections[i].close()
    for process in processes:
        process.join()

    # Convert the manager list to a regular list for easier access
    return results