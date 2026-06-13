def validate_extension(filename: str, allowed_exts=("csv",)) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_exts


def validate_size(size_bytes: int, max_bytes: int) -> bool:
    return size_bytes <= max_bytes
