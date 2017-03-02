import os

KHEPERA_LIB = os.environ.get('KHEPERA_LIB', None)

if KHEPERA_LIB is None:
    raise Exception('Environment variable KHEPERA_LIB is not set. Please set it to point to binaries of Khepera '
                    'simulation engine (see Installation section of Readme at https://github.com/Ewande/kheppy).')
