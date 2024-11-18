"""Line prefixers. Or suffixers, if you read RTL."""

from ansiwrap import ansilen

from .util import _PY2


class numberer(object):
    """
    Factory for numbering generators. Rather like Python 3's ``range`` (or
    Python 2's ``xrange``), but intended to number lines in a file so it uses
    natural numbers starting at 1, not the typical Python zero indexing. Returns
    formatted strings, not integers. Improves on what's possible as a functional
    generator on numberer because it can compute and return its own length, and
    be easily reset for multiple runs.
    """

    def __init__(self, start=1, template="{n:>3}: ", oneshot=True):
        """
        Make a numberer.
        """
        self.start = start
        self.n = start
        self.template = template
        self._formatted = None
        self.oneshot = oneshot

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next numbered template.
        """
        t = self.template
        if self._formatted:
            result = self._formatted
            self._formatted = None
        else:
            result = t.format(n=self.n)
        self.n += 1
        return result

    if _PY2:
        next = __next__

    def reset(self):
        if not self.oneshot:
            self.n = self.start

    def __len__(self):
        """
        What is the string length of the instantiated template now? NB This can change
        over time, as n does. Fixed-width format strings limit how often it can change
        (to when n crosses a power-of-10 boundary > the fixed template length
        can accommodate). This implementation saves the templated string it has created
        for reuse.
        """
        t = self.template
        result = t.format(n=self.n)
        self._formatted = result
        return ansilen(result)

    # TODO: Revisit strategy for prefix length compuation. Might be making it
    #       too complicated. Could just generate the string and then compute its
    #       ANSI length. To be absolutely, completely correct, would probably
    #       require a peekable generator front-end and a wrapping post-processor
    #       that checks results and invokes a rework of line M... for any line M
    #       that is found to have changed its prefix length from the prior.
    #       Because wrapping is not prefix-length savvy.


class first_rest(object):
    """
    Line prefixer (or suffixer) that gives one string for the first line, and
    an alternate string for every subsequent line. For implementing schemes
    like the Python REPL's '>>> ' followed by '... '.
    """
    def __init__(self, first, rest, oneshot=False):
        self.first = first
        self.rest = rest
        self.oneshot = oneshot
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = self.rest if self.counter else self.first
        self.counter += 1
        return result

    if _PY2:
        next = __next__

    def __len__(self):
        return ansilen(self.rest if self.counter else self.first)

    def reset(self):
        if not self.oneshot:
            self.counter = 0
