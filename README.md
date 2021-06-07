rdl-mid-to-gpkg

A utility program for GFDRR RDL to convert MapInfo files inside a 7z file or folder to geopackage.

Currently only tested on Windows.

This script assumes you have [7zip](https://www.7-zip.org/download.html) installed and available in PATH.

If unsure, see if this command returns an output with no error message.

```bash
$ 7z --help
```

# Usage

If the file is encrypted, set password as environment variable.

In a Windows terminal

```bash
$ set rdl_pw=SOME_PASSWORD
```

If using powershell:

```bash
$ $env:rdl_pw = 'SOME_PASSWORD'
```

Use with the `convert-to-gpkg` command

```bash
# WARNING: Temporary file will be created in current location.
$ python mid2gpkg.py convert-to-gpkg [PATH TO 7z FILE] --outdir [DESTINATION FOLDER]
```

If `--outdir` is not specified, then the files are placed in a `gpkgs` directory created in the
current location.


## Dev notes

**Environment setup**

Highly recommend using [`mamba`](https://mamba.readthedocs.io/en/latest/).

```bash
# Creating the environment
$ conda install -c conda-forge mamba
$ mamba env create -f environment.yml
```


**pip only approach**

```bash
$ pip install typer py7zr
```

Download and install python binaries

Download the files for your python version and pip install each in the order shown below

* https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
* https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj
* https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
* https://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona
* https://www.lfd.uci.edu/~gohlke/pythonlibs/#geopandas

```bash
# Example:
$ pip install GDAL-3.2.3-cp39-cp39-win_amd64.whl
```
