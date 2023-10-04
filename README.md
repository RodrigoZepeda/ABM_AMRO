# ABM_AMRO

Agent Based Model repository for antimicrobial resistance

![Figure of a collection of agents interacting](figures/abm_amro.svg)

## Setup

You can install the package via `pip`:

```bash
pip install git+https://github.com/RodrigoZepeda/ABM_AMRO
```

To install the development version:
```bash
pip install git+https://github.com/RodrigoZepeda/ABM_AMRO@dev
```


You can also download the repository and build locally with:
```bash
python setup.py build_ext -i
```

### Troubleshooting installation

#### OSX

Mac users require an installation of `libomp`. This can be done with homebrew via 
```bash
brew install libomp
```

If this is the first time you hear of homebrew you can check [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-homebrew-on-macos).

#### Ubuntu

Linux users require an installation of `openmp`. In Ubuntu you can do it with: 
```bash
apt install libomp-dev
```


#### Setting your compiler path for installation

You might have more than one compiler in your computer. The suggested ones are `gcc` for
Linux and Windows and `clang` for OSX. You can install using your
favourite compiler by passing the flag `CC` before pip:

```bash
CC=/path/to/your/gcc pip install git+https://github.com/RodrigoZepeda/ABM_AMRO
```

To find the path of your compiler you can do `where gcc` (or the compiler you want)
in Windows or `which gcc` (or the compiler you want) in Unix.

#### Request support
You can raise an issue to report installation issues. Make sure to include
the results of:
```bash
pip install --verbose git+https://github.com/RodrigoZepeda/ABM_AMRO
```

#### Windows (version >= 8 and 64 bit)


> :warning: I don't have Windows. I haven't tested it on Windows then. 

Apparently, you might need a `C++` compiler different from the native Windows one. You can download one
following [this instructions](https://code.visualstudio.com/docs/cpp/config-mingw#_prerequisites). If you are using R 
with `Rtools` you probably already have one (try `gcc --help`). 

The recommended way for Windows is to use the Windows Subsystem for Linux

#### Windows Subsystem for Linux

1. Go to Powershell and write:

```powershell
wsl --install
```

2. Download Ubuntu from the [Microsoft Store](https://apps.microsoft.com/store/detail/ubuntu/9PDXGNCFSCZV)

3. Open Ubuntu and follow the instructions entering a new username (all in lowercase letters)
```bash
Enter new UNIX username: your_username
```

4. Enter a password for the system
```bash
New password: [your_secret_password_invisibly]
```

5. Update the Ubuntu server

```bash
sudo apt update
sudo apt upgrade
```

6. Install other tools
```bash
sudo apt install build-essential git libomp-dev
```

7. Install miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

8. Exit the terminal and reopen it

```bash
exit
```

9. Create a new environment:
```bash
conda create -n AMRO
```


10. Go to the environment:
```bash
conda activate AMRO
```

11. Install pip:
```bash
conda install -y pip
```

12. Install the package!
```
pip install git+https://github.com/RodrigoZepeda/ABM_AMRO
```

## Developing on VSCODE for Windows WSL

1. Download the [WSL Extension](https://code.visualstudio.com/docs/remote/wsl) 
2. In VSCode press `F1` and choose `WSL: Connect to WSL using Distro in New Window`
3. Select the distro (`Ubuntu`)
4. Select the interpreter `CTRL + Shift + P > Python Select Interpreter > AMRO`
5. Start coding

### Add files
To add new files into your Ubuntu machine, from the WSL terminal do:
```bash
explorer.exe .
```
to open the Window's explorer and know where your virtual machine lies

### Install other packages 
From the `VSCode` terminal with `WSL` you can install packages with `conda` or `pip`
```
conda install pandas
pip install session_info
```

### Run Jupyter
1. Open a `ipynb` file and install [the extensions](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
2. You might also need to Select the interpreter via `CTRL + Shift + P > Python Select Interpreter > AMRO`



## Tutorials 

Go to the `tutorial/` folder for tutorials

## Contributing 

Go to **How to contribute** to learn more on the package's philosophy. 
