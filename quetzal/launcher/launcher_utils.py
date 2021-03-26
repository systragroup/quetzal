import os
import time
from subprocess import PIPE, Popen

import pandas as pd


def parallel_call_notebook(
    notebook,
    arg_list,
    stdout_path='log/out.txt',
    stderr_path='log/err.txt',
    workers=4,
    sleep=1,
    leave=False,
    errout_suffix=False
):
    os.system('jupyter nbconvert --to python %s' % notebook)
    file = notebook.replace('.ipynb', '.py')
    popens = {}

    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = arg if errout_suffix else ''
        suffix += '_' + notebook.split('.')[0]
        mode = 'w' if errout_suffix else 'a+'
        print(i, arg)
        with open(stdout_path.replace('.txt', '_' + suffix + '.txt'), mode) as stdout:
            with open(stderr_path.replace('.txt', '_' + suffix + '.txt'), mode) as stderr:
                popens[i] = Popen(
                    ['python', file, arg],
                    stdout=stdout,
                    stderr=stderr
                )
                if (i + 1) % workers == 0 or (i + 1) == len(arg_list):
                    # print('waiting')
                    popens[i].wait()
        time.sleep(sleep)
    if not leave:
        os.remove(file)
    for i in range(len(arg_list)):
        arg = arg_list[i]
        suffix = arg if errout_suffix else ''
        suffix += '_' + notebook.split('.')[0]
        mode = 'r'
        with open(stderr_path.replace('.txt', '_' + suffix + '.txt'), mode) as stderr:
            content = stderr.read()
            if 'Error' in content and "end_of_notebook" not in content:
                print("subprocess **{} {}** terminated with an error.".format(i, arg))
