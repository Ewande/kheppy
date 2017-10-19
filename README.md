# KhepPy

KhepPy stands for **Khep**era **Py**thon and is Python API with evolutionary computation and reinforcement learning algorithms for Khepera simulation engine. More information about the engine itself can be found [here](https://github.com/Ewande/khepera).

The idea behind KhepPy is to provide algorithms able to generate high-quality steering programs for Khepera robots.

## Getting started

The procedure to run KhepPy is fast and very easy.  

### Prerequisites

#### Python requirements
* Python 3
* NumPy
* (in future) Tensorflow

#### Khepera simulation engine binaries  
Build from source: [project page](https://github.com/Ewande/khepera).  
Or use precompiled binaries:
* [Linux](https://www.dropbox.com/s/dpcs0qsete8do2o/khepera_linux.so?dl=1) (tested on Ubuntu 14.04)
* [OS X/macOS](https://www.dropbox.com/s/1segnc3t6usninh/khepera_osx.so?dl=1) (tested on macOS Sierra/High Sierra)
* [Windows](https://www.dropbox.com/s/i4vvpkq4p5uu4c9/khepera_windows.dll?dl=1) (tested on Windows 7/8)

### Installing

1. Download or clone this repository to local directory of your choice.
2. Add main project directory to PYTHONPATH if you want to use KhepPy in external projects.  
   ```
   export PYTHONPATH="${PYTHONPATH}:/your/directory/kheppy"
   ```   
   Alternatively, place main project directory in *site-packages*.
3. Configure KHEPERA_LIB environment variable to point to Khepera simulation engine binaries.  
   ```
   export KHEPERA_LIB="/your/path/to/the/engine"
   ```  
   If you use PyCharm, go to *Run > Edit Configurations...* and add KHEPERA_LIB variable to *Environment variables* list.

### Test installation

For basic verification run:
```
python -c 'from kheppy.core import Simulation'
```
No output means kheppy.core should be ready to use.

## Examples
Now you can run some [examples](https://github.com/Ewande/kheppy/tree/master/examples) to familiarize yourself with KhepPy.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
