import re
import unicodedata
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

def convert_to_linux_filename(original_filename: str, replacement_char: str = '_') -> str:
    """
    Converts an arbitrary string to a valid Linux filename.

    This function aims to create a filename that is safe to use on Linux systems.
    It performs the following operations:
    1.  Replaces the NULL character ('\\0') and forward slash ('/') with the
        `replacement_char`. These are strictly disallowed in Linux filenames.
    2.  Replaces other characters that can be problematic in shells or are
        generally discouraged (e.g., `*`, `?`, `<`, `>`, `|`, `&`, `:`, `\\`,
        control characters, etc.) with the `replacement_char`.
    3.  Normalizes Unicode characters to a standard form (NFKC) to handle
        different representations of the same character and to decompose
        some ligatures or special characters into more basic forms.
    4.  Removes or replaces leading hyphens, as filenames starting with a
        hyphen can be misinterpreted as command-line options.
    5.  Replaces multiple occurrences of the `replacement_char` with a single
        instance.
    6.  Strips leading/trailing `replacement_char`s that might result from
        the sanitization, unless the entire filename becomes empty.
    7.  Ensures the filename is not empty. If sanitization results in an
        empty string, it returns a default name composed of the `replacement_char`.
    8.  While Linux filenames can technically be up to 255 bytes, this function
        does not currently enforce a strict length limit, but it's a consideration
        for more robust implementations.

    Args:
        original_filename: The input string to convert.
        replacement_char: The character to use for replacing invalid or
                          problematic characters. Defaults to '_'.

    Returns:
        A string that is a valid and safer Linux filename.
    """
    if not isinstance(original_filename, str):
        raise TypeError("Input 'original_filename' must be a string.")
    if not isinstance(replacement_char, str) or len(replacement_char) != 1:
        raise ValueError("'replacement_char' must be a single character string.")
    if replacement_char == '/' or replacement_char == '\0':
        raise ValueError("'replacement_char' cannot be '/' or the NULL character.")

    # 1. Normalize Unicode characters
    # NFKC is a good choice for compatibility and reducing "weird" characters
    # while trying to preserve the visual representation.
    filename = unicodedata.normalize('NFKC', original_filename)

    # 2. Replace NULL character and forward slash (strictly disallowed)
    filename = filename.replace('\0', replacement_char)
    filename = filename.replace('/', replacement_char)

    # 3. Define problematic characters (excluding '/')
    # This includes shell metacharacters, control characters (0-31),
    # and other characters that can cause issues.
    # ASCII control characters (0-31 or \x00-\x1F)
    # Other problematic characters: : \ * ? " < > | & $ ! ` ' ( ) [ ] { } ; # ^ % + = ~
    # We will create a regex pattern for these.
    # \x7F is DEL
    problematic_chars_pattern = r'[\\:\*\?"<>\|&\$!`\'\(\)\[\]\{\};#\^%\+=\~\x00-\x1F\x7F]'
    filename = re.sub(problematic_chars_pattern, replacement_char, filename)

    # 4. Handle leading hyphens
    if filename.startswith('-'):
        filename = replacement_char + filename[1:]

    # 5. Replace multiple occurrences of the replacement_char with a single instance
    if replacement_char: # Avoid issues if replacement_char is empty string (though disallowed by check)
        filename = re.sub(f'{re.escape(replacement_char)}+', replacement_char, filename)

    # 6. Strip leading/trailing replacement_char(s)
    # Only strip if the filename is not solely composed of replacement_char(s)
    stripped_filename = filename.strip(replacement_char)
    if stripped_filename:
        filename = stripped_filename
    elif not filename: # Original was empty or only invalid chars leading to empty string before this step
        filename = replacement_char # Default for completely empty/invalid input
    # If filename became a series of replacement_char and then stripped to empty,
    # it means original might have been e.g. "---". In this case, a single replacement_char is fine.
    # If filename was like "_abc_" and replacement is "_", strip gives "abc".
    # If filename was "___" and replacement is "_", strip gives "", so we ensure it's at least one replacement_char.
    if not filename: # if after stripping it's empty (e.g. input was "---" and replacement is "_")
         filename = replacement_char


    # 7. Ensure the filename is not empty
    if not filename:
        return replacement_char # Default filename if all characters were invalid

    # Note: Filename length limit (typically 255 bytes on Linux) is not strictly enforced here.
    # For a truly robust solution, you might want to truncate the filename:
    # max_len = 255
    # while len(filename.encode('utf-8')) > max_len:
    #     filename = filename[:-1]
    # (This naive truncation can break multi-byte characters, a better approach would be needed)

    return filename