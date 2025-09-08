import os
import sys
import psutil

def restart():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            print('closing open file: '+str(handler))
            os.close(handler.fd)
    except Exception as e:
        print('Error: '+e)

    python = sys.executable
    print('restart: '+str(python) + '::' + str(sys.argv))
    os.execl(python, python, *sys.argv)
