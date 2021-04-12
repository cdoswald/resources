# Resources

This repository contains tips, references, and code examples to help make programming easier!

## Virtual Environments

### Conda

Creating a new virtual environment:

1. Open Command Prompt
2. Update Conda with `conda update conda`
3. Create virtual environment with `conda create -n myenvname python=x.x anaconda`
4. Activate virtual environment with `conda activate myenvname`
5. Deactivate virtual environment with `conda deactivate`

Cloning a virtual environment: `conda create -n myclone --clone myenvname`

Installing packages in existing virtual environment: `conda install -n myenvname [packagename][=x.xx.x]`

Deleting a virtual environment: `conda remove -n myenvname -all`

Listing existing virtual environments: `conda info --envs`

## Improving Performance

### Measuring Performance

### Profiling Code

### PyPy

### Cython

