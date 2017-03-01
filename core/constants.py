import os

KHEPERA_LIB = os.environ.get('KHEPERA_LIB', None)

if KHEPERA_LIB is None:
    raise Exception('Environmental variable KHEPERA_LIB is not set. Please set it to point to shared object or DLL '
                    'with compiled simulation code from https://github.com/Ewande/khepera.')
