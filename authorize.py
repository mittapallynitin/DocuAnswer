import os


def is_authorized(user, password):
    _user = os.environ.get('USER')
    _password = os.environ.get('PASSWORD')
    return user == _user and password == _password
