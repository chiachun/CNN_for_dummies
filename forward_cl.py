import pyopencl as cl
from pyopencl import array
import numpy as np
import time
import os

def forward_fc_direct(x, pars, weights, bias) :
    
    (N, inPlane, outPlane, inSize) = pars
    

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1]
    context = cl.Context([device])
     
    # rRead in the OpenCL source file as a string
    f = open('forward_fc.cl', 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    #Allocate device memory and move input data from the host to the device memory
    mem_flags = cl.mem_flags
    
    
    x_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=x)
    w_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=weights)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=bias)
    
    out = np.zeros(N*outPlane).astype('float32')
    out_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out.nbytes)
     
    ts = time.time()
    complete = program.forward_fc_direct(queue, out.shape, None,
                                         inPlane, outPlane, inSize,
                                         x_buf, w_buf, b_buf, out_buf)
    events = [complete]
    
    cl.enqueue_copy(queue, out, out_buf, wait_for=events) # copy top_buf into top
    print "forward_direct: %f " % (time.time()-ts)
    return out


def forward_fc_block(x, pars, weights, bias) :
    
    (N, inPlane, outPlane, inSize) = pars
    

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1]
    context = cl.Context([device])
     
    # rRead in the OpenCL source file as a string
    f = open('forward_fc.cl', 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    #Allocate device memory and move input data from the host to the device memory
    mem_flags = cl.mem_flags
    
    
    x_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=x)
    w_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=weights)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=bias)
    
    out = np.zeros(N*outPlane).astype('float32')
    out_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out.nbytes)
     
    ts = time.time()
    complete = program.forward_fc_block(queue, (16,N), (16,1),
                                        inPlane, outPlane, inSize,
                                        x_buf, w_buf, b_buf, out_buf,
                                        cl.LocalMemory(256*4*8))
    events = [complete]
    
    cl.enqueue_copy(queue, out, out_buf, wait_for=events) # copy top_buf into top
    print "forward_fc: %f " % (time.time()-ts)
    return out



def forward_maxpool(x, pars):
    
    (N, inPlane, nFilter, filterSize, stride, padding, inSize, outSize) = pars
    # nFilter not used
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1]
    context = cl.Context([device])
     
    # rRead in the OpenCL source file as a string
    f = open('forward_maxpool.cl', 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    #Allocate device memory and move input data from the host to the device memory
    mem_flags = cl.mem_flags
    

    x_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=x)
   
    out = np.ones(N * inPlane * nFilter * outSize * outSize).astype('float32')
    sel = np.ones(N * inPlane * nFilter * outSize * outSize).astype('int32')
    
    out_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out.nbytes)
    sel_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, sel.nbytes)
   
    inSize = inSize 
    
    program.forward_maxpool(queue, out.shape, None,
                            inPlane, filterSize, stride, padding,
                            inSize, outSize, x_buf, out_buf, sel_buf)
    
    
    cl.enqueue_copy(queue, out, out_buf) # copy top_buf into top
    cl.enqueue_copy(queue, sel, sel_buf)
    return out, sel

def forward_conv_quick(x, pars, weights, bias) :
    
    (N, inPlane, nFilter, filterSize, stride, padding, inSize, outSize) = pars
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1]
    context = cl.Context([device])
     
    # rRead in the OpenCL source file as a string
    f = open('forward_conv.cl', 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    #Allocate device memory and move input data from the host to the device memory
    mem_flags = cl.mem_flags
    

    x_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=x)
    w_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=weights)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=bias)
    out = np.ones(N * inPlane * nFilter * outSize * outSize).astype('float32')
    out_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out.nbytes)
     
   
    inSize = inSize 
    
    program.forward_conv_quick(queue, out.shape, None,
                     inPlane, nFilter, filterSize, stride, padding,
                     inSize, outSize, x_buf, w_buf, b_buf, out_buf)
    
    
    cl.enqueue_copy(queue, out, out_buf) # copy top_buf into top
     
    return out





def forward_conv(x, pars, weights, bias) :
    
    (N, inPlane, nFilter, filterSize, stride, padding, inSize, outSize) = pars
    
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1]
    context = cl.Context([device])
     
    # rRead in the OpenCL source file as a string
    f = open('forward_conv.cl', 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    #Allocate device memory and move input data from the host to the device memory
    mem_flags = cl.mem_flags
    
    x_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=x)
    w_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=weights)
    b_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                      hostbuf=bias)
    out = np.ones(N * inPlane * nFilter * outSize * outSize).astype('float32')
    out_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out.nbytes)
     
    program.forward_conv(queue, out.shape, None,
                     inPlane, nFilter, filterSize, stride,
                     inSize, outSize, x_buf, w_buf, b_buf, out_buf)
    
    
    cl.enqueue_copy(queue, out, out_buf) # copy top_buf into top
     
    return out





