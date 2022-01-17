"""
Logging tools

Usage:
>>> from logger import *
>>> setLogger(__name__, DEBUG, INFO, or logging.WARNING) 
>>> error("some error occured.")
"""


import logging
import sys
from logging import FATAL,ERROR,WARNING,INFO,DEBUG,NOTSET

logger:logging.Logger = None
formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] | %(message)s"#,"%m-%d-%Y %H:%M:%S"
    )

def _checker(func):
    def check(*args, **kwds):
        global logger
        if logger is None:
            raise RuntimeError("Please call `setLogger` before logging.")
        func(*args,**kwds)
    return check

def setLogger(name:str,level = logging.WARN) -> None:
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    stream_hdlr = logging.StreamHandler(sys.stdout)
    stream_hdlr.setLevel(level)
    stream_hdlr.setFormatter(formatter)
    logger.addHandler(stream_hdlr)


def getLogger() -> logging.Logger:
    return logger

@_checker
def debug(*args, **kwds) -> None: logger.debug(*args,**kwds)
@_checker
def info(*args, **kwds) -> None: logger.info(*args,**kwds)
@_checker
def warning(*args, **kwds) -> None: logger.warning(*args,**kwds)
@_checker
def error(*args, **kwds) -> None: logger.error(*args,**kwds)
@_checker
def critical(*args, **kwds) -> None: logger.critical(*args,**kwds)
@_checker
def exception(*args, **kwds) -> None: logger.exception(*args,**kwds)
@_checker
def log(level,*args,**kwds) -> None: logger.log(level,*args,**kwds)

def _test():
    setLogger(__name__,logging.DEBUG)
    print("logger is",getLogger())

    debug("debug")

    info("info")


    warning("warning")

    error("error")

    critical("critical")
    
    exception("exception")

    #error
    global logger
    logger = None
    try:
        exception("_checker decorator is not working")
    except RuntimeError:
        print("_checker decorator is working")


if __name__ == "__main__":
    _test()