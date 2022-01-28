"""Miscellaneous utility function"""

import functools
from timeit import default_timer as timer

print_elapsed = False


def init_profile(do_profile):
    global print_elapsed
    print_elapsed = do_profile


def profile(func):
    @functools.wrappers(func)
    def wrapper(*args, **kwargs):
        global print_elapsed
        if not print_elapsed:
            return func(*args, **kwargs)
        begin = timer()
        res = func(*args, **kwargs)
        end = timer()
        print(f"Elapsed time for `{func.__name__}`: {end - begin}s.")
        return res

    return wrapper
