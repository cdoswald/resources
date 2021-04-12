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