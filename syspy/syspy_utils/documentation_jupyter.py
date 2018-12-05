# -*- coding: utf-8 -*-

__author__ = 'qchasserieau'


import os
import itertools
import json


def list_files(path, patterns):
    file_list = [os.path.join(path, file) for file in os.listdir(path) if file.split('.')[-1].lower() in patterns]
    subdirectories = [os.path.join(path, dir) for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    file_list += list(itertools.chain(*[list_files(subdirectory, patterns) for subdirectory in subdirectories]))
    return file_list


def makedoc(folder=os.getcwd(), doc_file=os.getcwd() + r'/doc.ipynb', old=None):

    """
    creates a .ipynb that gather all the documentation written in the firt cells of the notebooks of a given folder

    :param folder: the folder that contains the notebooks
    :type input_matrix: str
    :param doc_file: the file where the markdown cell of documentation will be included in 2nd position
    :type doc_file: str
    :param old: the classical folder name for old scripts that we do not want to include in the doc.
    :type old: str or None

    :return: None
    """

    files = [f.replace(folder, '') for f in list_files(folder, ['ipynb']) if 'checkpoint' not in f]
    text = ''

    for file in files:
        if old is None or old not in file:
            with open(folder + file, 'r', encoding='utf-8') as n:
                j = json.loads(n.read())

            text += '#### ' + file.replace('.ipynb', '') + '\n'
            try:
                if j['cells'][0]['cell_type'] == 'markdown':
                    text += ''.join(j['cells'][0]['source'])
                    text += '\n'
                else:
                    text += '\n'
            except IndexError:
                print(j, file)

    with open(doc_file, 'r', encoding='utf-8') as d:
        jd = json.loads(d.read())
    jd['cells'][1]['source'] = text

    with open(doc_file, 'w', encoding='utf-8') as d:
        d.write(json.dumps(jd))
