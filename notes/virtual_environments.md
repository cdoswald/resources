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

Within Command Prompt:

- Activate environment: `conda.bat activate <myenvname>`

- Deactivate environment: `conda deactivate`

### Spyder

Two methods for using a virtual environment within the Spyder IDE:

Method 1:

- Open Anaconda Prompt
- Activate environment
- Run Spyder from environment: `spyder --new-instance`
- If "spyder-kernels" error encountered, install spyder-kernels before running Spyder: `conda install spyder-kernels`

Method 2:

- Open Anaconda Prompt
- Activate environment
- Get the path for the Python executable within the environment: `where python` or `python -c "import sys; print(sys.executable)"`
- Within Spyder, go to Tools > Preferences > Python Interpreter
- Select "Use the following Python interpreter" and paste the path to the Python executable (in the virtual environment)
