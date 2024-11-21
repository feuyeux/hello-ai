# conda

<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

- **[Miniconda](https://docs.anaconda.com/free/miniconda/)** is a minimal installer provided by Anaconda. Use this installer if you want to install most packages yourself.
- **[Anaconda Distribution](https://www.anaconda.com/download)** is a full featured installer that comes with a suite of packages for data science, as well as Anaconda Navigator, a GUI application for working with conda environments.
- **[Miniforge](https://github.com/conda-forge/miniforge)** is an installer maintained by the conda-forge community that comes preconfigured for use with the conda-forge channel. To learn more about conda-forge, visit their website.

## Anaconda

### Mirrors

<https://repo.anaconda.com/archive/>
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/>

### [windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)

<https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Windows-x86_64.exe>

```env
PATH=D:\garden\anaconda3\Scripts
```

### [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

<https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh>

```sh
bash /mnt/c/Users/feuye/Downloads/Anaconda3-2024.06-1-Linux-x86_64.sh
```

### [Mac](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)

<https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-MacOSX-arm64.pkg>

### condarc

<https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/>

.condarc

```sh
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
```

### Check

```sh
$ conda info

     active environment : None
       user config file : /home/han/.condarc
 populated config files : /home/han/.condarc
          conda version : 24.5.0
    conda-build version : 24.5.1
         python version : 3.12.4.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=skylake
                          __conda=24.5.0=0
                          __cuda=12.5=0
                          __glibc=2.35=0
                          __linux=5.15.153.1=0
                          __unix=0=0
       base environment : /home/han/anaconda3  (writable)
      conda av data dir : /home/han/anaconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/noarch
          package cache : /home/han/anaconda3/pkgs
                          /home/han/.conda/pkgs
       envs directories : /home/han/anaconda3/envs
                          /home/han/.conda/envs
               platform : linux-64
             user-agent : conda/24.5.0 requests/2.32.2 CPython/3.12.4 Linux/5.15.153.1-microsoft-standard-WSL2 ubuntu/22.04.4 glibc/2.35 solver/libmamba conda-libmamba-solver/24.1.0 libmambapy/1.5.8 aau/0.4.4 c/. s/. e/.
                UID:GID : 1000:1000
             netrc file : None
           offline mode : False
```

```sh
conda info --verbose
```

## Miniconda

<https://docs.anaconda.com/miniconda/miniconda-install/>

```sh
$ bash Miniconda3-py312_24.5.0-0-MacOSX-x86_64.sh
$ conda info

     active environment : base
    active env location : /Users/hanl5/miniconda3
            shell level : 1
       user config file : /Users/hanl5/.condarc
 populated config files : /Users/hanl5/.condarc
          conda version : 24.5.0
    conda-build version : not installed
         python version : 3.12.4.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=skylake
                          __conda=24.5.0=0
                          __osx=10.16=0
                          __unix=0=0
       base environment : /Users/hanl5/miniconda3  (writable)
      conda av data dir : /Users/hanl5/miniconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/osx-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/osx-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/osx-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/noarch
          package cache : /Users/hanl5/miniconda3/pkgs
                          /Users/hanl5/.conda/pkgs
       envs directories : /Users/hanl5/miniconda3/envs
                          /Users/hanl5/.conda/envs
               platform : osx-64
             user-agent : conda/24.5.0 requests/2.32.2 CPython/3.12.4 Darwin/23.4.0 OSX/10.16 solver/libmamba conda-libmamba-solver/24.1.0 libmambapy/1.5.8 aau/0.4.4 c/. s/. e/.
                UID:GID : 501:20
             netrc file : None
           offline mode : False
```

## virtual env

### create

```sh
conda create -y --name tensorrt python=3.12
conda activate tensorrt

$ which python
/home/han/anaconda3/envs/tensorrt/bin/python
```

### remove

```sh
conda deactivate tensorrt
conda env remove -y -n tensorrt
conda env list
```
