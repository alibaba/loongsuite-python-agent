#!/usr/bin/env python3
"""
Build script: Package all required whl files into tar.gz

This script will:
1. Build all packages under instrumentation/
2. Build all packages under instrumentation-genai/
3. Build all packages under instrumentation-loongsuite/
4. Build util/opentelemetry-util-genai/
5. Skip duplicate packages according to config file
6. Package all whl files into tar.gz
"""

import argparse
import json
import logging
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Set, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_skip_config(config_path: Path) -> Set[str]:
    """Load package names to skip from config file"""
    if not config_path.exists():
        logger.warning(f"Config file {config_path} does not exist, using default config")
        return set()
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    skip_packages = set(config.get("skip_packages", []))
    logger.info(f"Loaded {len(skip_packages)} packages to skip from config file: {skip_packages}")
    return skip_packages


def get_package_name_from_whl(whl_path: Path) -> str:
    """Extract package name from whl filename"""
    # Format: package_name-version-py3-none-any.whl
    # Or: package_name-version-cp39-cp39-linux_x86_64.whl
    name = whl_path.stem  # Remove .whl
    # Find the part before the first number (version number) after the first -
    parts = name.split("-")
    if len(parts) >= 2:
        # Package name is all parts before version number, joined with -
        # Example: opentelemetry-instrumentation-langchain-2.0.0b0-py3-none-any
        # Package name is: opentelemetry-instrumentation-langchain
        # Need to find the position of version number (first part that looks like a version)
        package_parts = []
        for part in parts:
            # Version numbers usually contain digits and dots, or contain b0, dev, etc.
            if any(c.isdigit() for c in part) or part in ("dev", "b0", "b1", "rc0", "rc1"):
                break
            package_parts.append(part)
        return "-".join(package_parts)
    return name


def build_package(package_dir: Path, dist_dir: Path, existing_whl_files: Set[Path]) -> List[Path]:
    """Build whl file for a single package"""
    pyproject_toml = package_dir / "pyproject.toml"
    if not pyproject_toml.exists():
        logger.debug(f"Skipping {package_dir}, no pyproject.toml")
        return []
    
    logger.info(f"Building package: {package_dir}")
    try:
        # Record whl files before build
        before_whl_files = set(dist_dir.glob("*.whl"))
        
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Find newly generated whl files (exist after build but not before)
        after_whl_files = set(dist_dir.glob("*.whl"))
        new_whl_files = [f for f in after_whl_files - before_whl_files if f.suffix == ".whl"]
        
        if not new_whl_files:
            logger.warning(f"No new whl files found after building {package_dir}")
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr}")
        
        return sorted(new_whl_files)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build {package_dir}: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return []


def collect_packages(
    base_dir: Path,
    dist_dir: Path,
    skip_packages: Set[str],
) -> List[Path]:
    """Collect all packages that need to be built"""
    all_whl_files = []
    existing_whl_files = set(dist_dir.glob("*.whl"))
    
    # 1. Build packages under instrumentation/
    instrumentation_dir = base_dir / "instrumentation"
    if instrumentation_dir.exists():
        logger.info("Building packages under instrumentation/...")
        for package_dir in sorted(instrumentation_dir.iterdir()):
            if package_dir.is_dir() and (package_dir / "pyproject.toml").exists():
                whl_files = build_package(package_dir, dist_dir, existing_whl_files)
                all_whl_files.extend(whl_files)
                existing_whl_files.update(whl_files)
    
    # 2. Build packages under instrumentation-genai/
    instrumentation_genai_dir = base_dir / "instrumentation-genai"
    if instrumentation_genai_dir.exists():
        logger.info("Building packages under instrumentation-genai/...")
        for package_dir in sorted(instrumentation_genai_dir.iterdir()):
            if package_dir.is_dir() and (package_dir / "pyproject.toml").exists():
                whl_files = build_package(package_dir, dist_dir, existing_whl_files)
                all_whl_files.extend(whl_files)
                existing_whl_files.update(whl_files)
    
    # 3. Build packages under instrumentation-loongsuite/
    instrumentation_loongsuite_dir = base_dir / "instrumentation-loongsuite"
    if instrumentation_loongsuite_dir.exists():
        logger.info("Building packages under instrumentation-loongsuite/...")
        for package_dir in sorted(instrumentation_loongsuite_dir.iterdir()):
            if package_dir.is_dir() and (package_dir / "pyproject.toml").exists():
                whl_files = build_package(package_dir, dist_dir, existing_whl_files)
                all_whl_files.extend(whl_files)
                existing_whl_files.update(whl_files)
    
    # 4. Build util/opentelemetry-util-genai/
    util_genai_dir = base_dir / "util" / "opentelemetry-util-genai"
    if util_genai_dir.exists() and (util_genai_dir / "pyproject.toml").exists():
        logger.info("Building util/opentelemetry-util-genai/...")
        whl_files = build_package(util_genai_dir, dist_dir, existing_whl_files)
        all_whl_files.extend(whl_files)
        existing_whl_files.update(whl_files)
    
    # 5. Build loongsuite-distro/
    loongsuite_distro_dir = base_dir / "loongsuite-distro"
    if loongsuite_distro_dir.exists() and (loongsuite_distro_dir / "pyproject.toml").exists():
        logger.info("Building loongsuite-distro/...")
        whl_files = build_package(loongsuite_distro_dir, dist_dir, existing_whl_files)
        all_whl_files.extend(whl_files)
        existing_whl_files.update(whl_files)
    
    # 6. Filter out packages that need to be skipped
    filtered_whl_files = []
    skipped_count = 0
    seen_packages = {}  # Used to detect duplicate packages
    
    for whl_file in all_whl_files:
        package_name = get_package_name_from_whl(whl_file)
        
        # Check if in skip list
        if package_name in skip_packages:
            logger.info(f"Skipping package: {package_name} (according to config file)")
            skipped_count += 1
            # Delete skipped whl file
            whl_file.unlink()
            continue
        
        # Check for duplicate packages (same package may have multiple whl files, e.g., different platforms)
        if package_name in seen_packages:
            # Keep the newest file
            existing_file = seen_packages[package_name]
            if whl_file.stat().st_mtime > existing_file.stat().st_mtime:
                logger.debug(f"Replacing duplicate package {package_name}: {existing_file.name} -> {whl_file.name}")
                existing_file.unlink()
                seen_packages[package_name] = whl_file
                filtered_whl_files.remove(existing_file)
                filtered_whl_files.append(whl_file)
            else:
                logger.debug(f"Skipping older version {package_name}: {whl_file.name}")
                whl_file.unlink()
        else:
            seen_packages[package_name] = whl_file
            filtered_whl_files.append(whl_file)
    
    logger.info(f"Built {len(all_whl_files)} whl files in total")
    logger.info(f"Skipped {skipped_count} packages")
    logger.info(f"Final package contains {len(filtered_whl_files)} whl files")
    
    return filtered_whl_files


def create_tar_archive(whl_files: List[Path], output_path: Path):
    """Package all whl files into tar.gz"""
    logger.info(f"Creating tar archive: {output_path}")
    
    with tarfile.open(output_path, "w:gz") as tar:
        for whl_file in sorted(whl_files):
            # Only save filename, not path
            tar.add(whl_file, arcname=whl_file.name)
            logger.debug(f"Added to archive: {whl_file.name}")
    
    logger.info(f"Successfully created archive: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build loongsuite Python Agent release package"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory (default: script's parent directory)",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help="Build output directory (default: base-dir/dist)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "loongsuite-build-config.json",
        help="Config file path (default: scripts/loongsuite-build-config.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output tar.gz file path (default: dist/loongsuite-python-agent-<version>.tar.gz)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="dev",
        help="Version number (for output filename)",
    )
    
    args = parser.parse_args()
    
    base_dir = args.base_dir.resolve()
    dist_dir = args.dist_dir or (base_dir / "dist")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old whl files
    logger.info(f"Cleaning old build files: {dist_dir}")
    for old_file in dist_dir.glob("*.whl"):
        old_file.unlink()
    
    # Load skip config
    skip_packages = load_skip_config(args.config)
    
    # Collect and build all packages
    whl_files = collect_packages(base_dir, dist_dir, skip_packages)
    
    if not whl_files:
        logger.error("No whl files found, build failed")
        sys.exit(1)
    
    # Create tar archive
    output_path = args.output or (dist_dir / f"loongsuite-python-agent-{args.version}.tar.gz")
    create_tar_archive(whl_files, output_path)
    
    logger.info("Build completed!")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    main()

