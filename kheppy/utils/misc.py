from datetime import datetime as dt


def timestamp():
    now = dt.now()
    return '{}_{:02}_{:02}__{:02}_{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)


def to_str(collection, sep=' '):
    return sep.join(map(str, collection))
