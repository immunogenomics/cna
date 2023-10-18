import io
import sys

class DevNull(io.IOBase):
    def write(self, *args, **kwargs):
        pass
    
def select_output(allow=False):
    return sys.stdout if allow else DevNull()