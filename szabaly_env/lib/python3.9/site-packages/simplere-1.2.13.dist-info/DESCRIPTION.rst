
| |travisci| |version| |versions| |impls| |wheel| |coverage| |br-coverage|

.. |travisci| image:: https://travis-ci.org/jonathaneunice/simplere.svg?branch=master
    :alt: Travis CI build status
    :target: https://travis-ci.org/jonathaneunice/simplere

.. |version| image:: http://img.shields.io/pypi/v/simplere.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/simplere

.. |versions| image:: https://img.shields.io/pypi/pyversions/simplere.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/simplere

.. |impls| image:: https://img.shields.io/pypi/implementation/simplere.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/simplere

.. |wheel| image:: https://img.shields.io/pypi/wheel/simplere.svg
    :alt: Wheel packaging support
    :target: https://pypi.python.org/pypi/simplere

.. |coverage| image:: https://img.shields.io/badge/test_coverage-100%25-6600CC.svg
    :alt: Test line coverage
    :target: https://pypi.python.org/pypi/simplere

.. |br-coverage| image:: https://img.shields.io/badge/test_coverage-100%25-6600CC.svg
    :alt: Test branch coverage
    :target: https://pypi.python.org/pypi/simplere

A simplified interface to Python's regular expression (``re``) string
search. Eliminates steps and provides simpler access to results. As a bonus,
also provides compatible way to access Unix glob searches.

Usage
=====

Python regular expressions are powerful, but the language's lack
of an *en passant* (in passing) assignment requires a preparatory
motion and then a test::

    import re

    match = re.search(pattern, some_string)
    if match:
        print match.group(1)

With ``simplere``, you can do it in fewer steps::

    from simplere import *

    if match / re.search(pattern, some_string):
        print match[1]

That's particularly valuable in complex search-and-manipulate
code that requires multiple levels of searching along with
pre-conditions, error checking, and post-match cleanup, formatting,
and actions.

As a bonus,
``simplere`` also provides simple glob access.::

    if 'globtastic' in Glob('glob*'):
        print "Yes! It is!"
    else:
        raise ValueError('OH YES IT IS!')

It can also conveniently match against multiple glob
patterns, and/or do case-insensitive glob searches.

See `Read the Docs <http://simplere.readthedocs.org/en/latest/>`_
for the full installation and usage documentation.


