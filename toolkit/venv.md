# Python venv

## How to create a Python venv

### Python 3.4 and above

If you are running Python 3.4+, you can use the venv module baked into Python:

```sh
python -m venv <directory>
```

### All other Python versions

```sh
pip install virtualenv

virtualenv [directory]
```

## Python venv activation

### Windows

```sh
# In cmd.exe
<directory>\Scripts\activate.bat
# In PowerShell
<directory>\Scripts\Activate.ps1
```

### Linux and MacOS

```sh
source <directory>/bin/activate
```

## In vscode

[Using Python Environments in Visual Studio Code](https://code.visualstudio.com/docs/python/environments)
