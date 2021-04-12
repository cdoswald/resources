## Measuring Performance

### Timing Code
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

## Improving Performance

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

To generate an "HTML" file with the code annotation of Python interactions, pass `annotate=True` to `cythonize()` in *setup.py* file.
Alternatively, run `cython -a <script_name>.pyx` in the command prompt. The darker yellow a line of code is, the more Python interactions occur in that line.

Regular `.py` files can also be compiled with Cython using the *setup.py* file (pass to `cythonize()` function). Compiling the code could result in some increases in speed, even without Cython typing.

### PyPy

PyPy3 is installed as an executable file and can replace "Python.exe" when running scripts or starting interactive session. To install PyPy3 and additional packages:

1. (Recommended) Create and activate a virtual environment for PyPy
2. Download PyPy and unzip to any folder
3. Install ensurepip package: `<path-to-pypy-download>\pypy3 -m ensurepip --default-pip`
4. Install other packages: `<path-to-pypy-download>\pypy3 -mpip install <package_name>`
5. Start Python session: `<path-to-pypy-download>\pypy3` OR run script: `<path-to-pypy-download>\pypy3 <script-name>.py`

([StackOverflow Explanation](https://stackoverflow.com/a/49227568) for why PyPy make actually be slower for scripts that call numpy and pandas.)