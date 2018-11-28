import os
import shutil
import copy


class File:
    def __init__(self, soup):
        self.soup = soup
        self.soup_chunks = soup.split('\n')

    def __repr__(self):
        self.makesoup()
        return(self.soup)

    def makesoup(self):
        self.soup = '\n'.join([chunk for chunk in self.soup_chunks])


class Preface:
    def __init__(self, soup):
        self.soup = soup
        self.soup_chunks = soup.split('\n')

    def __repr__(self):
        self.makesoup()
        return(self.soup)

    def makesoup(self):
        self.soup = '\n'.join([chunk for chunk in self.soup_chunks])


class Program:

    def __init__(self, soup):
        self.soup = soup
        self.parse()

    def parse(self):
        self.preface = Preface(self.soup.split('#')[0])
        self.infiles = [File('#INFIL' + chunk) for chunk in self.soup.split('#OUTFIL')[0].split('#INFIL')[1:]]
        self.outfiles = [File('#OUTFIL' + chunk) for chunk in self.soup.split('#OUTFIL')[1:]]
        self.files = self.infiles + self.outfiles

    def __repr__(self):
        self.makesoup()
        return self.soup

    def makesoup(self):
        self.soup = str(self.preface) + ''.join([str(file) for file in self.files])


class Application:

    '''
    Contient toutes les informations caractéristiques d'une application
    '''

    def __init__(self, file_name, copy=False):
        self.file_name = file_name
        self.duplicata = file_name.split('.')[0]+'_duplicata_deleteme.'+file_name.split('.')[1]
        if copy:
            if not os.path.isfile(self.duplicata):
                shutil.copyfile(file_name, self.duplicata)
                print('copy')
        self.file = open( self.file_name, 'r')
        self.soup = self.file.read()
        self.file.close()
        self.eatsoup()

    def makesoup (self):
        self.soup = str(self.preface) + ''.join(['#PROGRAM' + str(program) for program in self.programs])

    def eatsoup(self):
        self.preface = Preface(self.soup.split('#PROGRAM')[0])
        self.programs = [Program(chunk) for chunk in self.soup.split('#PROGRAM')][1:]

    def __repr__(self):
        self.makesoup()
        return self.soup

    def dump(self):
        self.makesoup()
        self.file = open(self.file_name, 'w')
        self.file.write(self.soup)
        self.file.close()

    def addprogram(self, program):
        prog = copy.deepcopy(program)
        prog_id = len(self.programs)
        prog.preface.soup_chunks[0] = str(prog_id)  # on indique un id du programme qui suit ceux existants
        prog.preface.soup_chunks[3] = str(0)  # on fixe la priorité à 0
        prog.makesoup()
        self.programs.append(prog.soup)

    def addprograms(self, app):
        for program in app.programs:
            if 'addprograms' in program.preface.soup_chunks[12]:
                self.addprogram(program)

    def sortprograms(self):
        priority_zero_id = len(self.programs)
        for program in self.programs:
            if program.preface.soup_chunks[3]:  # si la priorité du programme est non nulle, on l'utilise pour numéroter le programme.
                program.preface.soup_chunks[0] = program.preface.soup_chunks[3]
            else:
                priority_zero_id +=1
                program.preface.soup_chunks[0] = priority_zero_id
        self.makesoup()
