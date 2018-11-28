__author__ = 'qchasserieau'
import re


class Line:
    def __init__(self, chunk):
        self.chunk = chunk
        try :
            self.name = self.chunk.split("'")[1]
        except:
            self.name = 'not_a_line'

    def change_time(self, time_factor=1, offset=0):
        self.chunk = _change_time(self.chunk, time_factor, offset)
        self.format_chunk()

    def add_line(self, line, start='left'):
        self.chunk = _add_chunk(self.chunk, line.chunk, start)
        self.format_chunk()

    def cut_at_node(self, n, keep='left'):
        self.chunk = _cut_at_node(self.chunk, n,  keep)
        self.format_chunk()

    def cut_between(self, from_node, to_node):
        #TODO understand and kill the fix below
        self.chunk = _cut_between(self.chunk, from_node,  to_node).replace('RT=-', 'RT=')
        self.format_chunk()

    def format_chunk(self):
        self.chunk = coma.sub(', ', equal.sub('=', self.chunk.replace('\n', ''))) + '\n'
        self.chunk = self.chunk.replace(', ,', ',')

    def formated_chunk(self):
        return coma.sub(', ', equal.sub('=', self.chunk.replace('\n', ''))) + '\n'

    def set_parameter(self, parameter, value):
        to_sub = re.compile(',[ ]*' + parameter + '[ ]*=[0-9TFtf]*')
        self.chunk = to_sub.sub(', %s=%s' % (parameter, str(value)), self.chunk)

    def drop_checkpoints(self):
        self.chunk = checkpoint_re.sub('', self.chunk)

    def set_direct(self, from_stop, to_stop):
        self.chunk = _set_direct(self.chunk, from_stop, to_stop)

    def change_stop(self, from_stop, to_stop):
        self.chunk = re.compile('N=[ ]*' + str(from_stop) + '[ ]*,').sub('N=' + str(to_stop) + ',', self.chunk)

    def __repr__(self):
        return self.chunk



equal = re.compile('[ ]*[=]+[ ]*')
coma = re.compile('[ ]*[,]+[ ]*')

checkpoint_re = re.compile('N=-[0-9]*,[ ]*RT=[0-9.]+[ ]*,')
regex_node_rt = 'N=[-]?[0-9]{4,6},[ ]?RT='
node_re = re.compile(regex_node_rt)
regex_time = 'RT=[0-9]{1,6}[.]?[0-9]{0,6}'
time_re = re.compile(regex_time)

def _stop_list(text, regex='N=[0-9]{4,6}'):
    stop_re = re.compile(regex)
    return [int(f[2:]) for f in stop_re.findall(text)]

def _add_chunk(left, right, start='left'):
    left_offset = _chunk_times(left)[-1]
    right_nodes_and_times = ', N=' + 'N='.join(_change_time(right, 1, left_offset).split('N=')[2:]) + '\n'
    if start == 'left':
        if _stop_list(left)[-1] == _stop_list(right)[0]:
            return left.replace('\n', '') + right_nodes_and_times
        else:
            print('terminus do not match : %i -- %i | %i -- %i' % (
                _stop_list(left)[0], _stop_list(left)[-1], _stop_list(right)[0], _stop_list(right)[-1]))
    if start == 'right':
        if _stop_list(right)[-1] == _stop_list(left)[0]:
            return left.split('N=')[0] + ', N=' + 'N='.join(_add_chunk(right, left, start='left').split('N=')[1:]) + '\n'
        else:
            print('terminus do not match : %i -- %i | %i -- %i' % (
                 _stop_list(right)[0], _stop_list(right)[-1]), _stop_list(left)[0], _stop_list(left)[-1])

def _chunk_times(chunk):
    clean_chunk = chunk.replace(' ', '')
    return [float(f[3:]) for f in time_re.findall(clean_chunk)]


def _chunk_nodes(chunk):
    clean_chunk = chunk.replace(' ', '')
    return [int(f[2:-4]) for f in node_re.findall(clean_chunk)]


def _change_time(chunk, time_factor=1, time_offset=0):
    clean_chunk = chunk.replace(' ', '')
    nodes_rt = [int(f[2:-4]) for f in node_re.findall(clean_chunk)]
    times = [float(f[3:]) for f in time_re.findall(clean_chunk)]
    _times = [round(t * time_factor + time_offset, 2) for t in times]
    t = ','.join(['N=' + str(node) + ',' + 'RT=' + str(time) for node, time in zip(nodes_rt, _times)])
    return chunk.split('N=')[0] + t + ' '


def _zip_rt_times(chunk, time_factor=1, time_offset=0):
    clean_chunk = chunk.replace(' ','')
    nodes_rt = [int(f[2:-4]) for f in node_re.findall(clean_chunk)]
    times = [float(f[3:]) for f in time_re.findall(clean_chunk)]
    _times = [round(t * time_factor + time_offset, 2) for t in times]
    return zip(nodes_rt, _times)


def _cut_at_node(chunk, n, keep='left'):
    s = 'N='+str(n)
    left = chunk.split('N=')[0]
    offset = -1*dict(_zip_rt_times(chunk))[n]

    if keep == 'right':
        return _change_time(left+ s + chunk.split(str(n))[1], 1, time_offset=offset)
    if keep == 'left':
        return chunk.split(s)[0] + s + ', RT=' + str(offset)


def _cut_between(chunk, na, nb, failed=False):
    try:
        test =  _cut_at_node(_cut_at_node(chunk, na, keep='right'), nb, keep='left')

        if (str(na) in chunk) and (str(nb) in test):
            return test
        else:
            return _cut_at_node(_cut_at_node(chunk, nb, keep='right'), na, keep='left')
    except:
        if failed == False:
            return _cut_between(chunk, nb, na, failed=True)
        else:
            return chunk


def _set_direct(chunk, from_stop, to_stop):
    try:
        a, b = str(from_stop), str(to_stop)
        regex = 'N=[ ]*' + a + '[ ]*,' + '.+' + 'N=[ ]*' + b + '[ ]*,'
        match = re.compile(regex).findall(chunk)[0]
        _chunk = chunk.replace(match, 'N='.join(match.split('N=')[:2]) + 'N=' + b + ', ')

    except IndexError:
        a, b = str(to_stop), str(from_stop)
        regex = 'N=[ ]*' + a + '[ ]*,' + '.+' + 'N=[ ]*' + b + '[ ]*,'
        match = re.compile(regex).findall(chunk)[0]
        _chunk = chunk.replace(match, 'N='.join(match.split('N=')[:2]) + 'N=' + b + ', ')

    return _chunk


