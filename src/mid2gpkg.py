import os, re, shutil
import subprocess
import zipfile
from glob import glob

import typer
import py7zr
from tqdm import tqdm

import numpy as np
import pandas as pd
# import fiona
# import fiona._shim
# import fiona.schema
import geopandas as gpd


try: 
    import zlib  # import necessary as it provides the compression
    compression = zipfile.ZIP_DEFLATED
except Exception:
    print("Could not load zlib library, any zipped files will not be compressed")
    compression = zipfile.ZIP_STORED


app = typer.Typer()

mid_pattern = re.compile(r"(.mid)")
mif_pattern = re.compile(r"(.mif)")


def _get_file_wo_ext(fn: str):
    """Get filename without extension."""
    return os.path.splitext(fn)[0]


def _get_file_list(src_file: str, pw: str):
    with py7zr.SevenZipFile(src_file, 'r', password=pw) as archive:
        allfiles = archive.getnames()

        # Identify mid/mif files
        mid_files = []
        mif_files = []
        for f in allfiles:
            if mid_pattern.search(f):
                mid_files.append(f)
            elif mif_pattern.search(f):
                mif_files.append(f)
    return mid_files, mif_files


def _attempt_remove(folder, num_attempts=10):
    """Attempt to remove a directory."""
    attempt_remove = True
    i = 0
    sep = os.path.sep if os.path.sep in folder else "/"
    base = folder.split(sep)[0]
    while attempt_remove and (i < num_attempts):
        try:
            shutil.rmtree(base)
            attempt_remove = False
        except FileNotFoundError:
            i += 1


# def zip_files(file_path: str):
#     """Zip a given gpkg file."""
#     fn = file_path

#     # Destination zip file
#     dest_zip = os.path.splitext(fn)[0] + ".zip"
#     (zipfile.ZipFile(dest_zip, mode='w')
#             .write(fn, compress_type=compression))
        

@app.command()
def list_files(src_file: str):
    """List all files in a given 7z file."""
    mid_files, mif_files = _get_file_list(src_file, pw)
    for mid, mif in zip(mid_files, mif_files):
        print(f"{mid} | {mif}")


@app.command()
def extract_files(src_file: str, filename:str = None, outdir:str = None):
    """Extract all files inside a 7z file."""
    command = f"7z x -p{pw} {src_file}"

    if filename:
        command += f" {filename}"
    
    if outdir:
        command += f" -o{outdir}"

    ret = subprocess.check_output(command).split()

    err_msg = f"Error extracting from {src_file}"
    if (b"No files to process" in ret):
        raise ValueError(err_msg)

    if (b"Everything is Ok" not in ret):
        raise ValueError(err_msg)


def convert_folder_to_gpkg(src_dir: str, outdir: str = "./gpkgs"):
    """Convert all MapInfo MID files inside the given directory to geopackage.
    
    Outputs a `report.csv` file with a summary of files found.
    """
    if src_dir.endswith("/"):
        src_dir = src_dir[:-1]

    fn_list = glob(src_dir+"/**/*.*", recursive=True)
    mid_files = []
    mif_files = []
    for f in fn_list:
        if mid_pattern.search(f):
            mid_files.append(f)
        elif mif_pattern.search(f):
            mif_files.append(f)

    # Extract mid/mif files and ensure valid
    report = pd.DataFrame({
        'filename': [],
        'mid_filesize': [],
        'mif_filesize': [],
        'num_features': [],
        'num_attributes': []
    })

    for idx, mid in tqdm(enumerate(mid_files)):
        pd_idx = idx

        base = _get_file_wo_ext(mid)

        try:
            mif = [f for f in mif_files if base in f][0]
        except IndexError:
            # no mif files...
            continue

        # Determine filesizes in MBs
        mid_fs = round(os.path.getsize(mid) / 1024**2, 4)
        mif_fs = round(os.path.getsize(mif) / 1024**2, 4)

        try:
            tmp = gpd.read_file(mid)

            # -1 col count for attributes as "geometries" will be one of the columns
            gpd_rows = len(tmp.index)
            gpd_cols = len(tmp.columns)-1

            # Coerce bytes to string if necessary
            for col in tmp.columns:
                if col == 'geometry':
                    continue
                if tmp[col].dtype == object:
                    if not np.all(tmp[col].apply(type) != bytes):
                        tmp[col] = tmp[col].apply(str)
            # End for

            # Convert to geopackage
            out_fn = _get_file_wo_ext(mid)
            out_fn = os.path.basename(out_fn)
            container = os.path.basename(os.path.dirname(base))
            os.makedirs(f"{outdir}/{container}", exist_ok=True)

            gpkg_file = f"{outdir}/{container}/{out_fn}.gpkg"
            tmp.to_file(gpkg_file, driver="GPKG")

            del(tmp)

            report.loc[pd_idx, :] = (base, mid_fs, mif_fs, gpd_rows, gpd_cols)

        except Exception as e:
            print(e)
            report.loc[pd_idx, :] = (f"{base}: {str(e)}", mid_fs, mif_fs, np.nan, np.nan)


@app.command()
def convert_to_gpkg(src_file: str, outdir: str = "./gpkgs"):
    """Convert all MapInfo MID files inside a 7z file to geopackage and zip them separately.
    
    Outputs a `report.csv` file with a summary of files found.

    WARNING: Temporary file will be created in current location.
    """
    if not src_file.endswith("7z"):
        if os.path.isdir(src_file):
            # assume directory if src_file does not have extension
            convert_folder_to_gpkg(src_dir=src_file, outdir=outdir)
        else:
            print("Specified file does not end with 7z, or is not a directory.")
        return

    mid_files, mif_files = _get_file_list(src_file, pw)

    command = "7z x"
    if pw is not None:
        command += f" -p{pw}"

    command += " {src_file} {fn}"

    # Extract mid/mif files and ensure valid
    report = pd.DataFrame({
        'filename': [],
        'mid_filesize': [],
        'mif_filesize': [],
        'num_features': [],
        'num_attributes': []
    })

    for idx, mid in tqdm(enumerate(mid_files)):
        pd_idx = idx

        base = _get_file_wo_ext(mid)
        err_msg = f"Error occurred extracting {base}"
        try:
            mif = [f for f in mif_files if base in f][0]
        except IndexError:
            # no mif files...
            continue

        try:
            flist = " ".join([mid, mif])

            ret = subprocess.run(command.format(src_file=src_file, fn=flist).split(),
                                    input="", capture_output=True)
            stdout = str(ret.stdout.decode('utf-8'))
            if ret.returncode != 0:
                if ("Archives with Errors" in stdout):
                    raise ValueError("Error extracting file: Corrupted, or incorrect password?")
                raise Exception()

            if ("No files to process" in stdout):
                raise ValueError(err_msg)

            if ("Everything is Ok" not in stdout):
                raise ValueError(err_msg)
        except ValueError as e:
            print(err_msg)
            # remove temporarily extracted files
            _attempt_remove(base)
            if "password" in str(e):
                raise e
            continue
        except Exception as e:
            # remove temporarily extracted files
            _attempt_remove(base)
            raise e

        # Determine filesizes in MBs
        mid_fs = round(os.path.getsize(mid) / 1024**2, 4)
        mif_fs = round(os.path.getsize(mif) / 1024**2, 4)

        try:
            tmp = gpd.read_file(mid)

            # -1 col count for attributes as "geometries" will be one of the columns
            gpd_rows = len(tmp.index)
            gpd_cols = len(tmp.columns)-1

            # Coerce bytes to string if necessary
            for col in tmp.columns:
                if col == 'geometry':
                    continue
                if tmp[col].dtype == object:
                    if not np.all(tmp[col].apply(type) != bytes):
                        tmp[col] = tmp[col].apply(str)
            # End for

            # Convert to geopackage
            out_fn = _get_file_wo_ext(mid)
            out_fn = os.path.basename(out_fn)
            container = os.path.basename(os.path.dirname(base))
            os.makedirs(f"{outdir}/{container}", exist_ok=True)

            gpkg_file = f"{outdir}/{container}/{out_fn}.gpkg"
            tmp.to_file(gpkg_file, driver="GPKG")

            # zip_files(gpkg_file)  # zip file up
            # os.remove(gpkg_file)  # delete originally created gpkg

            del(tmp)

            report.loc[pd_idx, :] = (base, mid_fs, mif_fs, gpd_rows, gpd_cols)

        except Exception as e:
            print(e)
            report.loc[pd_idx, :] = (f"{base}: {str(e)}", mid_fs, mif_fs, np.nan, np.nan)
        
        # remove temporarily extracted files
        _attempt_remove(base)

    report.to_csv("report.csv")


if __name__ == "__main__":
    pw = os.environ.get('rdl_pw', None)
    if pw is None:
        print("Note: No password set.")
    app()