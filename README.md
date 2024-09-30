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

You might need some additional libraries installed depending on your operating system:

### Troubleshooting installation

#### All operating systems

Make sure you are using numpy version 1.25

#### OSX

See the file `INSTALLING_OSX.md` for specific help

#### Ubuntu

Linux users require an installation of `openmp`. In Ubuntu you can do it with: 
```bash
apt install libomp-dev
```

#### Windows (version >= 8 and 64 bit)


> :warning: I don't have Windows. I haven't tested it on Windows then. 

Apparently, you might need a `C++` compiler different from the native Windows one. You can download one
following [this instructions](https://code.visualstudio.com/docs/cpp/config-mingw#_prerequisites). If you are using R 
with `Rtools` you probably already have one (try `gcc --help`). 

#### Setting your compiler path for installation

You might have more than one compiler in your computer. The suggested ones are `gcc` for
Linux and Windows and `clang` for OSX. You can install using your
favourite compiler by passing the flag `CC` before pip:

```bash
CC=/path/to/your/gcc pip install git+https://github.com/RodrigoZepeda/ABM_AMRO
```

To find the path of your compiler you can do `where gcc` (or the compiler you want)
in Windows or `which gcc` (or the compiler you want) in Unix.

#### Docker

This repository also provides a `Dockerfile` to run the repository in a container. To do so, from your terminal (OSX/Ubuntu) go to the folder with this repository and run:

```bash
docker build -t abm2024 .
```

to build the container. Then run the container:

```bash
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work abm2024
```

**Note** The `$(pwd):/home/jovyan/work` connects a volume from your computer to the docker. In the jupyter interface you'll see your files listed in the `work` directory.

Using your browser go to:
```
http://localhost:8888/
```

to open `docker`'s jupyter. To do that you'll also need a token that is automatically generated each time and listed in the terminal as jupyter runs. You can see it inside the `urls` listed. Here is mine where the token is `a11e01d7c327e403380c803e6c96c58ca36cc3e7ade89f38`:

![Image showing the terminal and a line with a url reading `http://127.0.0.1:8888/lab?token=a11e01d7c327e403380c803e6c96c58ca36cc3e7ade89f38` just below where it says Jupyter Server is running at...](figures/token.png)

Use that token to access Jupyter. 


#### Request support
You can raise an issue to report installation issues. Make sure to include
the results of:
```bash
pip install --verbose git+https://github.com/RodrigoZepeda/ABM_AMRO
```

## Tutorials 

Go to the `tutorial/` folder for tutorials

## Contributing 

Go to **How to contribute** to learn more on the package's philosophy. 
