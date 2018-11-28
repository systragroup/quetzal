from syspy.io.pandasdbf import dbf_qc as dbf
import xlrd
import tkinter as Tkinter
import os.path

_encoding = 'cp850'


def convert_value(value, encoding=_encoding):
    if isinstance(value, str):
        return value.encode(_encoding, 'replace')
    elif isinstance(value, float) and value % 1 == 0:
        return int(value)
    else:
        return value


class exceltodbf:
    def __init__(self, infile, outfile, encoding=_encoding):
        self.infile = infile
        self.wb = xlrd.open_workbook(infile)
        self.sheet_names = self.wb.sheet_names()
        self.mainWin = Tkinter.Tk()
        self.selectedSheet = Tkinter.StringVar(self.mainWin)
        self.selectedSheet.set(self.sheet_names[0])
        self.encoding = encoding
        self.convert(outfile)

    def convert(self, outfile ,event=None):
        if os.path.isfile(outfile):
            os.remove(outfile)

        sheet = self.wb.sheet_by_name(self.selectedSheet.get())
        fieldnames = [str(sheet.cell_value(0, col)) for col in range(sheet.ncols)]
        data = [[convert_value(sheet.cell_value(row, col), self.encoding) for col in range(sheet.ncols)] for row in range(1, sheet.nrows)]
        dbf.dbfwriter(outfile, fieldnames, data)
        self.exit()
    
    def exit(self):
        self.mainWin.destroy()

    