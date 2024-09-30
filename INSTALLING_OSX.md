# Installation on Mac (OSX)

The package uses a `C++` underneath. Currently (Sept. 2024) there seems to be an [issue](https://forums.developer.apple.com/forums/thread/738556) with the native Apple Clang compiler for version 15. 

> **Note** This instructions are for an Intel Mac. I don't know whether they are the same for a Silicon Mac. 

## Requirements

Mac users require an installation of `libomp`. This can be done with homebrew via 
```bash
brew install libomp
```

If this is the first time you hear of homebrew you can check [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-homebrew-on-macos).


## Installing the amro package with Hombrew's gcc

I suggest installing with Homebrew's `gcc`. Currently I have the package working with

```
g++-14 (Homebrew GCC 14.2.0) 14.2.0
gcc-14 (Homebrew GCC 14.2.0) 14.2.0
```

To install them:

```bash
brew install gcc@14
```

They will be somewhere in your `PATH`. In my case they are in:

```
/usr/local/bin/gcc-14
```

You can verify they are the appropriate versions by doing:

```bash
/usr/local/bin/gcc-14 -- version
/usr/local/bin/g++-14 --version
```

changing `/usr/local/bin/gcc-14` and `/usr/local/bin/g++-14` to your `PATH`. 

Finally, in your python environment do:

```bash
CC=/usr/local/bin/gcc-14 CXX=/usr/local/bin/g++-14 pip install --verbose git+https://github.com/RodrigoZepeda/ABM_AMRO@dev
```

And check that the `Test_2024.ipynb` works. 