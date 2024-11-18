"""
Home for separable utility functions used in say
"""

import itertools
import sys
import types
import codecs

# Basic Python version compatibility
_PY2 = sys.version_info[0] == 2
if _PY2:
    from StringIO import StringIO
    basestring = basestring   # so that can be exported
    unicode = unicode         # so that can be exported
    stringify = lambda v: v if isinstance(v, basestring) else unicode(v)
else:
    from codecs import getencoder
    basestring = unicode = str
    from io import StringIO
    stringify = str


def is_string(v):
    """
    Is the value v a string? Useful especially in making a test that works on
    both Python 2.x and Python 3.x
    """
    return isinstance(v, basestring)


def opened(f):
    """
    If f is a string, consider it a file path; return an open file that is ready
    to write to that path. Otherwise, assume it is an already open file and just
    return it. If it is a list or tuple (possibly of mixed strings / file paths
    and open files, do this action all each member of the list.

    Uses codecs.open to add auto-encoding in Python 2
    """
    if isinstance(f, (tuple, list)):
        return [ opened(ff) for ff in f ]
    if is_string(f):
        return codecs.open(f, mode='w', encoding='utf-8')
    return f


def flatten(*args):
    """
    Like itertools.chain(), but will pretend that single scalar values are
    singleton lists. Convenient for iterating over values whether they're lists
    or singletons.
    """
    flattened = [x if isinstance(x, (list, tuple)) else [x] for x in args]
    return itertools.chain(*flattened)

    # would use ``hasattr(x, '__iter__')`` rather than ``isinstance(x, (list, tuple))``,
    # but other objects like file have ``__iter__``, which screws things up


def next_str(g):
    """
    Given a generator g, return its next result as a unicode string. If not a
    generator, just return the value as a unicode string.
    """
    try:
        value = next(g)
    except TypeError:
        value = g if g is not None else ''
    return unicode(value)


def get_stdout():
    """
    Say objects previously had their own output encoding mechanism. It is now
    simplified, pushing encoding responsibility onto whatever underlying file
    object (or file analog) is being written to. While generally a good
    decision, it causes problems on some terminals (e.g. Komodo IDE) that for
    some reason initialize sys.stdout's encoding to US-ASCII. In those cases,
    instead of returning sys.stdout per se, return a writer object that does a
    rational encoding (UTF-8).
    """
    if sys.stdout.encoding == 'UTF-8': # pragma: no cover
        return sys.stdout              # true almost universally - but not under test
    else:
        return codecs.getwriter('UTF-8')(sys.stdout)
