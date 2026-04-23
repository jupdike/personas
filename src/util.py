from datetime import datetime


def timestamped_filename(stem: str, ext: str = "png") -> str:
    return f"{datetime.now():%Y-%m-%d-%H%M%S}-{stem}.{ext}"
