# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoongSuite Bootstrap Tool

Install all components of loongsuite Python Agent from tar.gz package.
Supports blacklist/whitelist to control which instrumentations to install.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Set, List, Tuple

from packaging.requirements import Requirement

logger = logging.getLogger(__name__)

# Base dependency packages (must be installed)
BASE_DEPENDENCIES = {
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation",
    "opentelemetry-util-genai",
    "opentelemetry-semantic-conventions",
}


def load_list_file(file_path: Path) -> Set[str]:
    """Load list from file (one package name per line)"""
    if not file_path.exists():
        return set()
    
    packages = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.add(line)
    
    return packages


def get_package_name_from_whl(whl_path: Path) -> str:
    """Extract package name from whl filename"""
    name = whl_path.stem
    parts = name.split("-")
    if len(parts) >= 2:
        package_parts = []
        for i, part in enumerate(parts):
            if any(c.isdigit() for c in part) or part in ("dev", "b0", "b1", "rc0", "rc1"):
                break
            package_parts.append(part)
        return "-".join(package_parts)
    return name


def download_file(url: str, dest: Path) -> Path:
    """Download file to specified path"""
    logger.info(f"Downloading file: {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Download completed: {dest}")
    return dest


def extract_tar(tar_path: Path, extract_dir: Path) -> List[Path]:
    """Extract tar.gz file, return all whl file paths"""
    logger.info(f"Extracting tar file: {tar_path} -> {extract_dir}")
    
    whl_files = []
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
        
        # Find all whl files
        for member in tar.getmembers():
            if member.name.endswith(".whl"):
                whl_path = extract_dir / member.name
                if whl_path.exists():
                    whl_files.append(whl_path)
    
    logger.info(f"Extraction completed, found {len(whl_files)} whl files")
    return sorted(whl_files)


def filter_packages(
    whl_files: List[Path],
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Filter packages based on blacklist/whitelist
    
    Returns:
        (base dependency packages list, instrumentation packages list)
    """
    base_packages = []
    instrumentation_packages = []
    
    blacklist = blacklist or set()
    whitelist = whitelist or set()
    
    for whl_file in whl_files:
        package_name = get_package_name_from_whl(whl_file)
        
        # Check blacklist
        if blacklist and package_name in blacklist:
            logger.debug(f"Skipping package (blacklist): {package_name}")
            continue
        
        # Check whitelist
        if whitelist and package_name not in whitelist:
            logger.debug(f"Skipping package (not in whitelist): {package_name}")
            continue
        
        # Classify: base dependencies vs instrumentation
        if package_name in BASE_DEPENDENCIES:
            base_packages.append(whl_file)
        else:
            instrumentation_packages.append(whl_file)
    
    return base_packages, instrumentation_packages


def install_packages(whl_files: List[Path], find_links_dir: Path, upgrade: bool = False):
    """Install packages using pip"""
    if not whl_files:
        logger.warning("No packages to install")
        return
    
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--find-links",
        str(find_links_dir),
    ]
    
    if upgrade:
        cmd.append("--upgrade")
    
    # Add all whl files
    cmd.extend([str(whl) for whl in whl_files])
    
    logger.info(f"Executing install command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Installation completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        raise


def install_from_tar(
    tar_path: Path,
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    upgrade: bool = False,
    keep_temp: bool = False,
):
    """
    Install loongsuite packages from tar package
    
    Args:
        tar_path: tar file path or URL (can be Path or str)
        blacklist: blacklist (do not install these packages)
        whitelist: whitelist (only install these packages if specified)
        upgrade: whether to upgrade already installed packages
        keep_temp: whether to keep temporary directory
    """
    # If it's a URL, download first
    tar_path_str = str(tar_path)
    if tar_path_str.startswith(("http://", "https://")):
        temp_tar = Path(tempfile.mkdtemp()) / "loongsuite.tar.gz"
        download_file(tar_path_str, temp_tar)
        tar_path = temp_tar
    else:
        tar_path = Path(tar_path)
    
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file does not exist: {tar_path}")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="loongsuite-"))
    
    try:
        # Extract tar file
        whl_files = extract_tar(tar_path, temp_dir)
        
        if not whl_files:
            raise ValueError("No whl files found in tar file")
        
        # Filter packages
        base_packages, instrumentation_packages = filter_packages(
            whl_files, blacklist, whitelist
        )
        
        # Ensure base dependencies must be installed
        if not base_packages:
            logger.warning("Warning: No base dependency packages found, this may cause installation to fail")
        
        # Merge all packages to install
        all_packages = base_packages + instrumentation_packages
        
        logger.info(f"Will install {len(base_packages)} base dependency packages")
        logger.info(f"Will install {len(instrumentation_packages)} instrumentation packages")
        
        # Install
        install_packages(all_packages, temp_dir, upgrade)
        
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            logger.info(f"Temporary directory kept at: {temp_dir}")


def get_latest_release_url(repo: str = "alibaba/loongsuite-python-agent") -> str:
    """Get latest release tar.gz URL from GitHub API"""
    import urllib.request
    import json as json_lib
    
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    logger.info(f"Fetching latest release: {api_url}")
    
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json_lib.loads(response.read())
            for asset in data.get("assets", []):
                if asset["name"].endswith(".tar.gz"):
                    return asset["browser_download_url"]
        
        # If no asset found, try to build URL from tag
        tag = data.get("tag_name", "").lstrip("v")
        return f"https://github.com/{repo}/releases/download/{data.get('tag_name')}/loongsuite-python-agent-{tag}.tar.gz"
    except Exception as e:
        logger.error(f"Failed to fetch latest release: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="""
        LoongSuite Bootstrap - Install loongsuite Python Agent from tar package
        
        This tool installs all loongsuite components from tar.gz file.
        Supports blacklist/whitelist to control which instrumentations to install.
        """
    )
    
    parser.add_argument(
        "-t",
        "--tar",
        type=Path,
        help="tar package path or GitHub Releases URL",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="version number, download from GitHub Releases (e.g., 1.0.0)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="install latest version (from GitHub Releases)",
    )
    parser.add_argument(
        "--blacklist",
        type=Path,
        help="blacklist file path (one package name per line, do not install these packages)",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        help="whitelist file path (one package name per line, only install these packages)",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="upgrade already installed packages",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep temporary directory (for debugging)",
    )
    parser.add_argument(
        "-a",
        "--action",
        choices=["install", "requirements"],
        default="install",
        help="action type: install to install packages, requirements to output package list",
    )
    
    args = parser.parse_args()
    
    # Determine tar file path
    tar_path = None
    if args.tar:
        tar_path = args.tar
    elif args.version:
        tar_path = f"https://github.com/alibaba/loongsuite-python-agent/releases/download/v{args.version}/loongsuite-python-agent-{args.version}.tar.gz"
    elif args.latest:
        tar_path = get_latest_release_url()
    else:
        parser.error("Must specify one of --tar, --version, or --latest")
    
    # Load blacklist/whitelist
    blacklist = load_list_file(args.blacklist) if args.blacklist else None
    whitelist = load_list_file(args.whitelist) if args.whitelist else None
    
    if blacklist:
        logger.info(f"Blacklist: {len(blacklist)} packages")
    if whitelist:
        logger.info(f"Whitelist: {len(whitelist)} packages")
    
    if args.action == "requirements":
        # Output package list
        tar_path_str = str(tar_path)
        if tar_path_str.startswith(("http://", "https://")):
            temp_tar = Path(tempfile.mkdtemp()) / "loongsuite.tar.gz"
            download_file(tar_path_str, temp_tar)
            tar_path = temp_tar
        else:
            tar_path = Path(tar_path)
        
        temp_dir = Path(tempfile.mkdtemp(prefix="loongsuite-"))
        try:
            whl_files = extract_tar(tar_path, temp_dir)
            base_packages, instrumentation_packages = filter_packages(
                whl_files, blacklist, whitelist
            )
            
            print("# LoongSuite Python Agent Package List")
            print("# Base dependency packages (must be installed):")
            for whl in base_packages:
                package_name = get_package_name_from_whl(whl)
                print(f"{package_name}")
            
            print("\n# Instrumentation packages:")
            for whl in instrumentation_packages:
                package_name = get_package_name_from_whl(whl)
                print(f"{package_name}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Install
        install_from_tar(
            tar_path,
            blacklist=blacklist,
            whitelist=whitelist,
            upgrade=args.upgrade,
            keep_temp=args.keep_temp,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    main()


