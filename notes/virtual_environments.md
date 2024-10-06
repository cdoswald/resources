## Virtual Environments

### Conda

Within Anaconda Prompt:

- Update Conda: `conda update conda`

- List existing virtual environments: `conda info --envs` or `conda env list`

- List packages in virtual environment: `conda list`

- Create new virtual environment: `conda create -n [env_name]`

- Create new virtual environment with Python version/Anaconda distribution: `conda create -n [env_name] python=[x.x] anaconda`

- Clone existing environment: `conda create -n myclone --clone [env_name]`

- Activate environment: `conda activate [env_name]`

- Deactivate environment: `conda deactivate`

- Install packages in existing environment: `conda install -n [env_name] [packagename]=[x.xx.x]`

- Install packages (and dependencies) from a particular channel: `conda install -c [channel-name] [packagename]`

- Show Conda channels: `conda config --show channels`

- Add Conda channel/move existing channel to highest priority: `conda config --add channels [channel-name]`

- Add Conda channel to bottom of priority list: `conda config --append channels [channel-name]`

- Set strict priority so only first channel is used: `conda config --set channel_priority strict`

- Show Conda config: `conda config --show`

- Delete environment: `conda remove -n [env_name] -all`

- Export environment as YAML: `conda env export > environment.yml`

- Create environment from YAML: `conda env create -f environment.yml`

    - First line of YAML file determines name of new environment

Within Command Prompt:

- Activate environment: `conda.bat activate [env_name]`

- Deactivate environment: `conda deactivate`

Pip and Conda

- [Best Practices](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)

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
