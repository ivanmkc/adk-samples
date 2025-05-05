import re
from pathlib import Path


def get_next_run_directory(base_dir: Path) -> Path:
    """
    Finds the next available directory name in the format 'run_i'
    within the given base directory.

    Args:
        base_dir: A pathlib.Path object representing the directory
                  where 'run_i' directories are located or will be created.

    Returns:
        A pathlib.Path object representing the next directory to create
        (e.g., base_dir / 'run_0', base_dir / 'run_1', etc.).
    """
    base_dir.mkdir(parents=True, exist_ok=True)  # Ensure base_dir exists

    max_i = -1
    run_dir_pattern = re.compile(r"^run_(\d+)$")

    for item in base_dir.iterdir():
        if item.is_dir():
            match = run_dir_pattern.match(item.name)
            if match:
                current_i = int(match.group(1))
                if current_i > max_i:
                    max_i = current_i

    next_i = max_i + 1
    next_run_dir_name = f"run_{next_i}"
    next_dir = base_dir / next_run_dir_name

    next_dir.mkdir(parents=True, exist_ok=True)  # Ensure base_dir exists

    return next_dir
