import datetime
import decimal
import struct


def dbfreader(f):
    """Returns an iterator over records in a Xbase DBF file.

    The first row returned contains the field names.
    The second row contains field specs: (type, size, decimal places).
    Subsequent rows contain the data records.
    If a record is marked as deleted, it is skipped.

    File should be opened for binary reads.

    """
    # See DBF format spec at:
    #     http://www.pgts.com.au/download/public/xbase.htm#DBF_STRUCT

    numrec, lenheader = struct.unpack('<xxxxLH22x', f.read(32))
    numfields = (lenheader - 33) // 32

    fields = []
    for fieldno in xrange(numfields):
        name, typ, size, deci = struct.unpack('<11sc4xBB14x', f.read(32))
        name = name.replace('\0', '')       # eliminate NULs from string
        fields.append((name, typ, size, deci))
    yield [field[0] for field in fields]
    yield [tuple(field[1:]) for field in fields]

    terminator = f.read(1)
    assert terminator == '\r'

    fields.insert(0, ('DeletionFlag', 'C', 1, 0))
    fmt = ''.join(['%ds' % fieldinfo[2] for fieldinfo in fields])
    fmtsiz = struct.calcsize(fmt)
    for i in xrange(numrec):
        record = struct.unpack(fmt, f.read(fmtsiz))
        if record[0] != ' ':
            continue                        # deleted record
        result = []
        for (name, typ, size, deci), value in zip(fields, record):
            if name == 'DeletionFlag':
                continue
            if typ == "N":
                value = value.replace('\0', '').lstrip()
                # if value == '':
                #     value = 0
                if value == '**********':  # null?
                    value = ''
                elif deci:
                    value = decimal.Decimal(value)
                else:
                    value = int(value)
            elif typ == 'D':
                y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                value = datetime.date(y, m, d)
            elif typ == 'L':
                value = (value in 'YyTt' and 'T') or (value in 'NnFf' and 'F') or '?'
            result.append(value)
        yield result


def dbfwriter(*args):
    """ If 4 arguments given, same as dbfwriter_raw below
    The other usage is :
        dbfwriter(filepath, fieldnames, data)
    this will cause a dbfwriter_raw call with appropriate fieldspecs (and file opening)
    """
    if len(args) == 4:
        dbfwriter_raw(args[0], args[1], args[2], args[3])
    elif len(args) == 3:
        def getspecs(value):
            if isinstance(value, (int, float, decimal.Decimal)):
                if '.' in str(value):
                    totallen = len(str(value))
                    floatlen = len(str(value).split('.')[1])
                else:
                    totallen = len(str(value))
                    floatlen = 0
                return ('N', totallen, floatlen)
            else:
                return ('C', len(str(value)), 0)

        def maxspecs(spec1, spec2):
            if spec1[0] == spec2[0] == 'N':
                totallen = max(spec1[1] - spec1[2], spec2[1] - spec2[2]) + max(spec1[2], spec2[2])
                floatlen = max(spec1[2], spec2[2])
                return ('N', totallen, floatlen)
            else:
                return ('C', max(spec1[1], spec2[1]), 0)

        fieldspecs = [('N', 0, 0)] * len(args[2][0])
        for record in args[2]:
            for i, value in enumerate(record):
                fieldspecs[i] = maxspecs(fieldspecs[i], getspecs(value))
        dbfwriter_raw(open(args[0], 'wb'), args[1], fieldspecs, args[2])


def dbfwriter_raw(f, fieldnames, fieldspecs, records):
    """ Return a string suitable for writing directly to a binary dbf file.

    File f should be open for writing in a binary mode.

    Fieldnames should be no longer than ten characters and not include \x00.
    Fieldspecs are in the form (type, size, deci) where
        type is one of:
            C for ascii character data
            M for ascii character memo data (real memo fields not supported)
            D for datetime objects
            N for ints or decimal objects
            L for logical values 'T', 'F', or '?'
        size is the field width
        deci is the number of decimal places in the provided decimal object
    Records can be an iterable over the records (sequences of field values).
    """
    # header info
    ver = 3
    now = datetime.datetime.now()
    yr, mon, day = now.year - 1900, now.month, now.day
    numrec = len(records)
    numfields = len(fieldspecs)
    lenheader = numfields * 32 + 33
    lenrecord = sum(field[1] for field in fieldspecs) + 1
    hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec, lenheader, lenrecord)
    f.write(hdr)

    # field specs
    for name, (typ, size, deci) in zip(fieldnames, fieldspecs):
        name = name.ljust(11, '\x00')
        fld = struct.pack('<11sc4xBB14x', name.encode('cp850'), typ.encode('cp850'), size, deci)
        f.write(fld)

    # terminator
    f.write('\r'.encode('cp850'))

    # records
    for record in records:
        f.write(bytes(' ', 'cp850'))                        # deletion flag
        for (typ, size, deci), value in zip(fieldspecs, record):
            if typ == "N":
                value = str(value).rjust(size, ' ')
            elif typ == 'D':
                value = value.strftime('%Y%m%d')
            elif typ == 'L':
                value = str(value)[0].upper()
            else:
                value = str(value.decode('cp850'))[:size].ljust(size, ' ') if type(value) == bytes else str(value)[:size].ljust(size, ' ')
            assert len(value) == size
            f.write(bytes(str(value), 'cp850'))

    # End of file
    f.write('\x1A'.encode('cp850'))
