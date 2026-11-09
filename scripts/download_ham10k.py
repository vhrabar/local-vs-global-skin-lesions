"""
Download the HAM10k dataset from Harvard db into a given directory.

Usage:
    uv run scripts/download_ham10000.py path/to/dir --with-test --with-segmentations
"""

import hashlib
import json
import zipfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

DATAVERSE_URL = "https://dataverse.harvard.edu"
DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"

CORE_FILES = {"HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip", "HAM10000_metadata.tab"}
TEST_FILES = {"ISIC2018_Task3_Test_Images.zip", "ISIC2018_Task3_Test_GroundTruth.tab"}
SEGMENTATION_FILES = {"HAM10000_segmentations_lesion_tschandl.zip"}

CHUNK_SIZE = 1 << 20
HEADERS = {"User-Agent": "ds-downloader/0.1"}


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Download the HAM10k ds")
    parser.add_argument("output_dir", type=Path, help="download dir")
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--with-test", action="store_true", help="also download the ISIC2018 Task 3 test set")
    parser.add_argument("--with-segmentations", action="store_true", help="also download lesion segmentation masks")
    parser.add_argument("--keep-archives", action="store_true", help="also keep zip files")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with (
        path.open("rb") as f,
        tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc=f"verify {path.name}", leave=False) as bar,
    ):
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def is_valid(path: Path, size: int, md5: str | None) -> bool:
    if not path.exists() or path.stat().st_size != size:
        return False
    return md5 is None or md5sum(path) == md5


def fetch(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (
        urlopen(Request(url, headers=HEADERS)) as response,
        tmp.open("wb") as f,
        tqdm(total=size, unit="B", unit_scale=True, desc=f"download {dest.name}", leave=False) as bar,
    ):
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def extract(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX/")]
        nested = all(m.startswith(f"{archive_path.stem}/") for m in members)
        target = output_dir if nested else output_dir / archive_path.stem
        for member in tqdm(members, unit="file", desc=f"extract {archive_path.name}", leave=False):
            archive.extract(member, target)


def download(output_dir: Path, wanted: set[str], force: bool, keep_archives: bool) -> None:
    metadata_url = f"{DATAVERSE_URL}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    with urlopen(Request(metadata_url, headers=HEADERS)) as response:
        files = [f["dataFile"] for f in json.load(response)["data"]["latestVersion"]["files"]]
    files = [f for f in files if f["filename"] in wanted]

    for data_file in tqdm(files, unit="file", desc="HAM10k"):
        url = f"{DATAVERSE_URL}/api/access/datafile/{data_file['id']}"
        if "originalFileName" in data_file:
            url += "?format=original"
            name = Path(data_file["originalFileName"]).with_suffix(".csv").name
            size, md5 = data_file["originalFileSize"], None
        else:
            name = data_file["filename"]
            size, md5 = data_file["filesize"], data_file["checksum"]["value"]

        dest = output_dir / name
        extracted_marker = output_dir / f".{name}.extracted"
        if not force and (extracted_marker.exists() or is_valid(dest, size, md5)):
            tqdm.write(f"skipping {name}, already present")
        else:
            fetch(url, dest, size)
            if not is_valid(dest, size, md5):
                dest.unlink()
                raise RuntimeError(f"checksum mismatch for {name}")
            extracted_marker.unlink(missing_ok=True)

        if dest.suffix == ".zip" and not extracted_marker.exists():
            extract(dest, output_dir)
            extracted_marker.touch()
            if not keep_archives:
                dest.unlink()

    tqdm.write(f"HAM10000 downloaded to: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(CORE_FILES)
    if args.with_test:
        wanted |= TEST_FILES
    if args.with_segmentations:
        wanted |= SEGMENTATION_FILES
    download(output_dir, wanted, args.force, args.keep_archives)


if __name__ == "__main__":
    main()
