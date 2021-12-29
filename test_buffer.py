import numpy as np
from buffer import buffer
import sys

stderr = sys.__stderr__

def test(buf_kind):
    stderr.write('+---------------+\n')
    stderr.write('|    Testing    |\n')
    stderr.write('+---------------+\n')

    with_cuda = 1 if buf_kind == 'CUDA' else 0

    stderr.write('---- creating %s buffer ... \n'%(buf_kind))
    buf = buffer(16, 3.1415, with_cuda)
    stderr.write('---- done!\n\n')

    stderr.write('---- sharing with numpy ... \n')
    nparr = np.array(buf, copy=False)
    stderr.write('---- done!\n\n')

    stderr.write('---- display numpy ... \n')
    stderr.write('nparr.__array_interface__ = %s\n'%(str(nparr.__array_interface__)))
    stderr.write('nparr = %s\n'%(str(nparr)))
    stderr.write('---- done!\n\n')

    stderr.write('---- destroy numpy\n')
    nparr = None
    stderr.write('---- done!\n\n')


    stderr.write('---- destroy buffer ... \n')
    buf = None
    stderr.write('---- done!\n\n')

    stderr.write('\n\n\n\n')



#test('CPU')

test('CUDA')

