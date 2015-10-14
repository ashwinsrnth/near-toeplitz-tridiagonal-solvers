import jinja2
nx = 128
ny = 128
nz = 128
by = 1

with open('kernels.jinja2') as f:
    kernel_src = f.read()

tpl = jinja2.Template(kernel_src)
kernel = tpl.render(nx=nx, ny=ny, nz=nz, by=by)

with open('kernels.cu', 'w') as f:
    f.write(kernel)

    
