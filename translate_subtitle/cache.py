from functools import wraps
from pathlib import Path


def cache(dir_: Path):
    dir_.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def with_caching(*args, **kwargs):
            file_name = str(kwargs["_cache_file"])
            file_path = dir_ / file_name
            if not file_path.exists():
                ret = func(*args, **kwargs)
                with file_path.open(mode="w") as f:
                    f.write(ret)
                return ret

            with file_path.open() as f:
                return f.read()

        return with_caching

    return decorator
