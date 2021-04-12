# Resources

This repository contains tips, references, and code examples to help make programming easier!

## Virtual Environments

### Conda

Within Anaconda Prompt:

- Update Conda: `conda update conda`

- List existing virtual environments: `conda info --envs`

- Create new virtual environment: `conda create -n myenvname python=x.x anaconda`

- Clone existing environment: `conda create -n myclone --clone myenvname`

- Activate environment: `conda activate myenvname`

- Deactivate environment: `conda deactivate`

- Install packages in existing environment: `conda install -n myenvname [packagename][=x.xx.x]`

- Delete environment: `conda remove -n myenvname -all`

## Improving Performance

### Measuring Performance

```
from timeit import default_timer as timer

start_time = timer()

# Code

end_time = timer()
print(f'Elapsed time: {round(end_time - start_time, 2)} seconds')
```

### Profiling Code

```
import cProfile
import pstats

import local_module

cProfile.runctz("local_module.function(args)", globals(), locals(), "profile.prof")
with open("profiler_report.txt", "w") as stream:
    ps = pstats.Stats("profile.prof", stream=stream)
    ps.strip_dirs().sort_stats("time", "cumulative").print_stats()
    ps.strip_dirs().sort_stats("time", "cumulative").print_callers() # Optional
```

### Cython

To compile and run Cython code, first write Python code in a ".pyx" file. Then, compile one of two ways:

Option 1: Use pyximport package to skip creating a *setup.py* file (for basic use cases):
```
import pyximport; pyximport.install()
import local_module
```

Option 2: Create *setup.py* file and run `python setup.py build_ext --inplace` via Command Prompt. The *setup.py* file should look like the following:
```
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='<name of application>',
    ext_modules=cythonize("<python_script_name>.pyx"),
    zip_safe=False
)
```

### PyPy

1. Download [PyPy build](https://www.pypy.org/) 
2. Unzip files to any folder
3. (Recommended) Create and activate virtual environment for PyPy
4. Start Python session: `<path-to-pypy-download>\pypy3` OR run script: `<path-to-pypy-download>\pypy3 <script-name>.py`
