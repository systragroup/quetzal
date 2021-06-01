import itertools
import os

from syspy.pycube.application import Application


def multiple_replace(text, word_dict):
    for key in word_dict:
        text = text.replace(key, word_dict[key])
    return text


def list_files(path, patterns):
    files = [os.path.join(path, file) for file in os.listdir(path) if file.split('.')[-1].lower() in patterns]
    subdirectories = [os.path.join(path, dir) for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    files += list(itertools.chain(*[list_files(subdirectory, patterns) for subdirectory in subdirectories]))
    return files


def list_apps(path):
    apps = [os.path.join(path, app) for app in os.listdir(path) if (app.endswith('.app') or app.endswith('.APP'))]
    subdirectories = [os.path.join(path, dir) for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    apps += list(itertools.chain(*[list_apps(subdirectory) for subdirectory in subdirectories]))
    return apps


def replace_in_file(filename, old, new):
    with open(filename, 'r') as file:
        soup = file.read()
    count = soup.count(old)
    with open(filename, 'w') as file:
        file.write(soup.replace(old, new))
    return count


class Project:
    """
    Object for processing text, scripts an apps in a cube project, based on top of pycube.application.
    """
    def __init__(self, catalog_dir):
        self.catalog_dir = catalog_dir
        self.apps = [Application(app_file) for app_file in list_apps(self.catalog_dir)]
        self.cube_generated_text_like_files = list_files(
            self.catalog_dir, patterns=['app', 'txt', 's', 'vpr', 'cpl', 'py'])

    def change_catalog_dir(self, old_dir, new_dir='default'):

        """
        :param old_dir: current path to the catalog directory as written in the scripts and apps
        :type old_dir: str
        :param new_dir: new path to the catalog directory to be written in the scripts and apps
        :type new_dir: str
        :return: None
        """
        _new_dir = self.catalog_dir if new_dir == 'default' else new_dir
        occurrences = 0
        for filename in self.cube_generated_text_like_files:
            occurrences += replace_in_file(filename, old_dir, _new_dir)

        print(str(occurrences) + ' replacement have been performed ')

    def replace(self, old_string, new_string):

        """
        :param old_string: string to replace
        :type old_dir: str
        :param new_string: new string
        :type new_dir: str
        :return: None
        """
        occurrences = 0
        for filename in self.cube_generated_text_like_files:
            occurrences += replace_in_file(filename, old_string, new_string)

        print(str(occurrences) + ' change(s)')
