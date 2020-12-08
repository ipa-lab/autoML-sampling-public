# this file was copied from: https://github.com/josepablocam/ams/tree/master/experiments
import multiprocessing as mp
from multiprocessing.context import TimeoutError
import sys

MP_INITIALIZED = False


def init_mp():
    # change start method to avoid issues with crashes/freezes
    # discussed in
    # http://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    global MP_INITIALIZED
    if MP_INITIALIZED:
        return

    try:
        print("Setting mp start method to forkserver")
        mp.set_start_method('forkserver')
        MP_INITIALIZED = True
    except (RuntimeError, ValueError) as err:
        # already set
        pass


def run(seconds, fun, *args, **kwargs):
    if seconds > 0:
        pool = mp.Pool(processes=1)
        try:
            proc = pool.apply_async(fun, args, kwargs)
            result = proc.get(seconds)
            return result
        finally:
            pool.terminate()
            pool.close()
    else:
        # if no timeout, then no point
        # in incurring cost of running as separate process
        # so call locally
        return fun(*args, **kwargs)
