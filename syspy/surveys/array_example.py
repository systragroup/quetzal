import pandas as pd
import json


def build_json():
    with open('orthogonal_arrays.txt') as file:
        text = file.read()
    chunks = text[1:].split('\n\n\n')
    
    arrays = []

    for chunk in chunks:

        name = chunk.split('\n')[0]

        lines = chunk.split('\n')[1:]
        lines = [line.replace(' ', '') for line in lines if line]
        array = [[int(char) for char in line] for line in lines ]

        exponents = name.split('n')[0]
        runs = int(name.split('n=')[-1])

        array_dict = {
            'array': json.dumps(array),
            'runs': runs
        }
        exp_dict = {}

        for exp_group in exponents.split(' '):
            try: 
                key, exp = exp_group.split('^')
                exp_dict[int(key)] = int(exp)
            except ValueError: # ValueError
                pass
        array_dict.update(exp_dict)
        array_dict['exponents'] = exp_dict
        arrays.append(array_dict)


    base = pd.DataFrame([pd.Series(array) for array in arrays]).fillna(0)
    int_columns = sorted([c for c in base.columns if type(c) is int])
    string_columns = [c for c in base.columns if type(c) is str]
    base[int_columns] = base[int_columns].astype(int)

    base = base[string_columns + int_columns] 

    base.sort_values(['runs'] + int_columns, inplace=True)
    base.reset_index(inplace=True, drop=True)
    base['len'] = base['array'].apply(lambda s: len(json.loads(s)))
    base['error'] = (base['len'] - base['runs'])
    
    with open('orthogonal_arrays.json', 'w') as file:
        file.write(base.to_json(orient='records'))


def try_int(s):
    try: 
        return int(s)
    except ValueError: 
        return s

import os
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    global base
    with open(dir_path + r'/orthogonal_arrays.json', 'r') as file:
        base = pd.read_json(file.read())
        
    base['exponents'] = base['exponents'].apply(
        lambda d: {int(k): v for k, v in d.items()}
    )
    base.columns = [try_int(c) for c in base.columns]

main()

