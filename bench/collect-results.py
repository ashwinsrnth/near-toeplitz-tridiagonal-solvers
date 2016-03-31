import subprocess
import os

sizes = [32, 64, 128, 256]

for size in sizes:
    nrhs = size**2
    p = subprocess.Popen(["./bench-neato.py", str(size), str(nrhs), "--use_shmem"], stdout=subprocess.PIPE)
    t1 = float(p.communicate()[0])
    p = subprocess.Popen(["./bench-neato.py", str(size), str(nrhs)], stdout=subprocess.PIPE)
    t2 = float(p.communicate()[0])
    p = subprocess.Popen(["cusparse/bench-cusparse", str(size), str(nrhs)], stdout=subprocess.PIPE)
    t3 = float(p.communicate()[0])
    os.environ["OMP_NUM_THREADS"] = "1"
    p = subprocess.Popen(["mkl/bench-mkl", str(size), str(nrhs)], stdout=subprocess.PIPE, env=os.environ)
    t4 = float(p.communicate()[0])
    os.environ["OMP_NUM_THREADS"] = "2"
    p = subprocess.Popen(["mkl/bench-mkl", str(size), str(nrhs)], stdout=subprocess.PIPE, env=os.environ)
    t5 = float(p.communicate()[0])
    os.environ["OMP_NUM_THREADS"] = "4"
    p = subprocess.Popen(["mkl/bench-mkl", str(size), str(nrhs)], stdout=subprocess.PIPE, env=os.environ)
    t6 = float(p.communicate()[0])
    os.environ["OMP_NUM_THREADS"] = "8"
    p = subprocess.Popen(["mkl/bench-mkl", str(size), str(nrhs)], stdout=subprocess.PIPE, env=os.environ)
    t7 = float(p.communicate()[0])
    os.environ["OMP_NUM_THREADS"] = "16"
    p = subprocess.Popen(["mkl/bench-mkl", str(size), str(nrhs)], stdout=subprocess.PIPE, env=os.environ)
    t8 = float(p.communicate()[0])

    print t1, t2, t3, t4, t5, t6, t7, t8
