from unittest import TestCase


class MyTest(TestCase):
    # basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '\\data'
    # uri = ':memory:'
    # #uri = os.path.join(basedir, 'app.db')
    # SQLALCHEMY_DATABASE_URI = "sqlite:///" + uri
    # TESTING = True

    def setUp(self):
        pass

    def tearDown(self):
        pass
        # db.session.remove()
        # db.drop_all()
