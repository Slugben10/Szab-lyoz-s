"""Interpolating string formatter. """

import string
import inspect
import sys
import os
import re
from options import Options, OptionsContext, Transient, Prohibited
from options.chainstuf import chainstuf
from options.methset import *
import six
import ansiwrap
from ansiwrap import ansilen
from say.util import *
from say.util import _PY2
from say.styling import *
from say.vertical import Vertical, vertical
from say.version import __version__

### Workhorse functions

sformatter = string.Formatter()  # piggyback Python's format() template parser

QUOTES = ("'", '"')

def populate_style(style_name, styles):
    if style_name.startswith(QUOTES):
        style_name = style_name.strip(style_name[0])
    if style_name not in styles:
        styles[style_name] = autostyle(style_name)


def _sprintf(arg, caller, styles=None, override=None):
    """
    Format the template string (arg) with the values seen in the context of the
    caller. If override is defined, it is a Mapping providing additional values
    atop those of the local context.
    """

    def seval(s):
        """
        Evaluate the string s in the caller's context. Return its value.
        """
        try:
            localvars = caller.f_locals if override is None \
                                        else chainstuf(override, caller.f_locals)
            return eval(s, caller.f_globals, localvars)
        except SyntaxError:
            raise SyntaxError("syntax error when formatting '{0}'".format(s))

    def parse_style(fs, styles):
        """
        Get the style component out of the format string, if any.
        """
        if "style" in fs:
            fs_parts = fs.split(',')
            raw_fs_parts = []
            for fsp in fs_parts:
                if fsp.startswith('style='):
                    style_name = fsp[6:].strip(QUOTE_DELIM_STR)
                else:
                    raw_fs_parts.append(fsp)
            populate_style(style_name, styles)
            return style_name, raw_fs_parts
        else:
            return None, fs

        # TODO: Replace this parser with something more professional
        # TODO: join=

    if is_string(arg):
        arg = unicode(arg) if _PY2 and isinstance(arg, str) else arg
        parts = []
        for (literal_text, field_name, format_spec, conversion) in sformatter.parse(arg):
            parts.append(literal_text)
            if field_name is not None:

                style_name, raw_format_spec = parse_style(format_spec, styles)
                format_str = six.u("{0") + \
                                   ("!" + conversion if conversion else "") + \
                                   (":" + raw_format_spec if raw_format_spec else "") + "}"
                field_value = seval(field_name)
                formatted = format_str.format(field_value)
                if style_name and style_name in styles:
                    formatted = styles[style_name](formatted)

                parts.append(formatted)
        return ''.join(parts)
    else:
        return str(seval(str(arg)))

def extended_len(x):
    """
    Executable prefixes and suffixes like numberers complicate life compared to simple
    strings, because they are generators, and the length of their returned strings is
    not particularly knoable.
    """
    return ansilen(x) if is_string(x) else len(x)


@enable_method_set
class Say(object):
    """
    Say provides high-level printing functions. Instances are configurable
    and callable.
    """

    options = Options(
        indent=0,           # indent level (if set to None, indentation is turned off)
        indent_str='    ',  # indent string for each level
        prefix='',          # prefix each line with this (string or string generator)
        suffix='',          # suffix each line with this (string or string generator)
        files=[get_stdout()], # where is output headed? a list of write() able objects
        file=Transient,     # file to write to - experimental compatibility trick
        wrap=None,          # column to wrap lines to, if any
        sep=' ',            # separate args with this (Python print function compatible)
        vsep=None,          # vertical separation
        end='\n',           # end output with this (Python print function compatible)
        silent=False,       # be quiet
        style=None,         # name of style in which to wrap entire output line
        styles={},          # style dict
        _callframe=Transient, # frame from which the caller was calling
    )

    options.magic(
        indent = lambda v, cur: cur.indent + int(v) if isinstance(v, (str, unicode, Relative)) else v,
        files  = lambda v, cur: opened(v)
    )

    def __init__(self, **kwargs):
        """
        Make a Say instance with the given options.
        """
        self.options = Say.options.push(kwargs)
        self.__version__ = __version__


    @staticmethod
    def escape(s):
        """
        Double { and } characters in a string to 'escape' them so ``str.format``
        doesn't treat them as template characters. NB This is NOT idempotent!
        Escaping more than once (when { or } are present ) = ERROR.
        """
        return s.replace('{', '{{').replace('}', '}}')

    @method_set
    def hr(self, **kwargs):
        """
        Print a horizontal line. Like the HTML hr tag. Optionally
        specify the width, character repeated to make the line, and vertical separation.

        Good options for the separator may be '-', '=', or parts of the Unicode
        box drawing character set. http://en.wikipedia.org/wiki/Box-drawing_character
        """
        opts = method_push(self.options, self.hr.__kwdefaults__, kwargs)

        # no interpolation required, so no caller frame introspection
        line = opts.char * opts.width
        return self._output(line, opts)

    @method_set
    def title(self, name, **kwargs):
        """
        Print a horizontal line with an embedded title.
        """
        opts = method_push(self.options, self.title.__kwdefaults__, kwargs)
        opts.setdefault('_callframe', inspect.currentframe().f_back)

        formatted = _sprintf(name, opts._callframe, opts.styles) if is_string(name) else str(name)
        bars = opts.char * opts.width
        line = ' '.join([bars, formatted, bars])
        return self._output(line, opts)

    @method_set
    def sep(self, text='', **kwargs):
        """
        Print a short horizontal line, possibly with some text following,
        of the desired width. Useful as a separator for different parts
        of output.
        """
        opts = method_push(self.options, Say.sep.__kwdefaults__, kwargs)
        opts.setdefault('_callframe', inspect.currentframe().f_back)
        formatted = _sprintf(text, opts._callframe, opts.styles) if is_string(text) else stringify(text)
        bars = opts.char * opts.width
        line = ' '.join([bars, formatted]) if formatted else bars
        return self._output(line, opts)

    @method_set
    def blank_lines(self, n, **kwargs):
        """
        Output N blank lines ("vertical separation"). Unlike other methods, this
        does not obey normal vertical separation rules, because it is about
        explicit vertical separation. If it obeyed vsep, it would usually gild
        the lily (double space).
        """
        if not n:
            return
        opts = method_push(self.options, Say.blank_lines.__kwdefaults__, kwargs)
        opts.vsep = None
        # no interpolation required, so no caller frame introspection
        return self._output([''] * n, opts)

    def set(self, **kwargs):
        """
        Permanently change the reciver's settings to those defined in the kwargs.
        An update-like function.
        """
        self.options.set(**kwargs)

    def setfiles(self, files):
        """
        Set the list of output files. ``files`` is a list. For each item, if
        it's a real file like ``sys.stdout``, use it. If it's a string, assume
        it's a filename and open it for writing.
        """
        self.options.files = [opened(f) for f in files]
        return self

        # TBD: Turn this into 'magical' attribute set

    def settings(self, **kwargs):
        """
        Open a context manager for a `with` statement. Temporarily change settings
        for the duration of the with.
        """
        return SayContext(self, kwargs)

    def clone(self, **kwargs):
        """
        Create a new instance whose options are chained to this instance's
        options (and thence to self.__class__.options). kwargs become the
        cloned instance's overlay options.
        """
        cloned = self.__class__()
        cloned.options = self.options.push(kwargs)
        return cloned

    but = clone

    def fork(self, **kwargs):
        """
        Create a new instance whose options are chained to this instance's
        class's options, then this instance's current values, then any
        difference values stated in kwwargs.
        """
        forked = self.__class__()
        forked.options = forked.options.push({})
        for k, v in self.options.items():
            if v != forked.options[k]:
                forked.options[k] = v
        forked.options = forked.options.push(kwargs)  # not previously processed
        return forked

    # should determine variations of cloning
    # How much dynamic linkage should clone have to its predecessor?
    # Full dynamic linkage along the clone/but model? Or all of the
    # parent's options as they now are, frozen in time, then building
    # on those? Propose "fork()" as a "fork in the road" kind of diversion
    # in which the Class options + instance options as they now stand +
    # kwargs are in place. Thus common heritage and common values to start,
    # but no ongoing linkage.

    def style(self, *args, **kwargs):
        """
        Define a style.
        """
        for k,v in kwargs.items():
            if isinstance(v, six.string_types):
                self.options.styles[k] = autostyle(v)
            else:
                self.options.styles[k] = v

    def verbatim(self, *args, **kwargs):
        """
        Say, but without interpretation. Useful for just its text decoration
        features.
        """
        opts = self.options.push(kwargs)
        opts.setdefault('_callframe', inspect.currentframe().f_back)

        formatted = [ unicode(arg) for arg in args ]
        return self._output(opts.sep.join(formatted), opts)

    def __call__(self, *args, **kwargs):
        """
        Primary interface. say(something)
        """
        opts = self.options.push(kwargs)
        opts.setdefault('_callframe', inspect.currentframe().f_back)

        formatted = [ _sprintf(arg, opts._callframe, opts.styles) if is_string(arg) else unicode(arg)
                      for arg in args ]
        return self._output(opts.sep.join(formatted), opts)

    def _return_value(self, outstr, opts):
        """
        Prepare a quality return value. Manages encoding of string and
        stripping of final newline, if desired, based on opts.return_encoded
        and opts.return_strip_newline.
        """
        ret_encoding = opts.encoding if opts.return_encoded is True else opts.encoded
        ret_encoded = outstr.encode(ret_encoding) if ret_encoding else outstr
        if opts.return_strip_newline:
            ret_encoded = ret_encoded[:-1] if ret_encoded.endswith('\n') else ret_encoded
        return ret_encoded

    def _outstr(self, data, opts):
        """
        Given result text, format it. ``data`` may be either a
        list of lines, or a composed string. NB: Don't feed it a list of strings,
        some of which contain newlines; that will break its assumptions.
        """
        datalines = data if isinstance(data, list) else data.splitlines()

        if opts.indent or opts.wrap or opts.prefix or opts.suffix or opts.vsep or opts.style:
            indent_str = opts.indent_str * opts.indent
            if opts.wrap:
                datastr = '\n'.join(datalines)
                # compute number of characters left for payload wrapping
                # to fit without desired wrap length
                prefix_len = extended_len(opts.prefix)
                suffix_len = extended_len(opts.suffix)
                indent_len = len(indent_str)
                margin_len = prefix_len + suffix_len + indent_len
                wrap_width = opts.wrap - margin_len
                wrappedlines = ansiwrap.wrap(datastr,
                                             width=wrap_width,
                                             replace_whitespace=False,
                                             initial_indent='',
                                             subsequent_indent='')
                datalines = []
                for line in wrappedlines:
                    datalines.extend(line.splitlines())
            if opts.style:
                styler = opts.styles.get(opts.style, None)
                if not styler:
                    styler = opts.styles.setdefault(opts.style, autostyle(opts.style))
                datalines = [styler(line) for line in datalines]
            if opts.indent:
                datalines = [ indent_str + line for line in datalines ]
            vbefore, vafter = vertical(opts.vsep).render()
            datalines = vbefore + datalines + vafter
            if opts.prefix or opts.suffix:
                datalines = [''.join([next_str(opts.prefix), line, next_str(opts.suffix)])
                              for line in datalines]
            outstr = '\n'.join(datalines)
        else:
            outstr = '\n'.join(data) if isinstance(data, list) else data
        # by end of indenting, dealing with string only

        # prepare and emit output
        if opts.end is not None:
            outstr += opts.end
        return outstr

    def _output(self, data, opts):
        """
        Construct the output string and write it to files.
        """
        if opts.prefix and hasattr(opts.prefix, 'reset'):
            opts.prefix.reset()
        if opts.suffix and hasattr(opts.suffix, 'reset'):
            opts.suffix.reset()
        if opts.silent:
            return
        else:
            outstr = self._outstr(data, opts)
            if opts.file:
                files = [opts.file]
            else:
                files = opts.files
            for f in files:
                f.write(outstr)


class SayContext(OptionsContext):
    """
    Context helper to support Python's with statement.  Generally called
    from ``with say.settings(...):``
    """
    pass


class SayReturn(Say):

    """
    Combo of Say and Fmt that both says and returns data. Not a typical
    use case, but consistent with original design of ``Say``, and needed
    for ``show`` module
    """

    options = Say.options.add(
        retvalue=True,     # return formatted value if this is so
        trimrv=True,       # remove end (usually "\n") from retval if so
    )

    def __init__(self, **kwargs):
        self.options = SayReturn.options.push(kwargs)
        self.options.styles = say.options.styles  # styles are idiosyncratially shared

    def _output(self, data, opts):
        """
        Construct the output string, write it to files, and return it.
        """
        outstr = self._outstr(data, opts)
        if opts.trimrv and opts.end and outstr.endswith(opts.end):
            retstr = outstr[:-len(opts.end)]
        else:
            retstr = outstr

        if not opts.silent:
            outstr = self._outstr(data, opts)
            for f in opts.files:
                f.write(outstr)

        return retstr


class Fmt(Say):

    """
    A type of Say that returns its result, rather than writes it
    to files.
    """

    options = Say.options.copy().add(
        files=Prohibited,   # no files needed
        end=None,           # no extra newline
        silent=Prohibited,  # Fmt is never silent
    )

    def __init__(self, **kwargs):
        self.options = Fmt.options.push(kwargs)
        self.options.styles = say.options.styles  # styles are idiosyncratially shared


    def _output(self, data, opts):
        """
        Construct the output string and return it.
        """
        return self._outstr(data, opts)


### Define default callables
say = Say()
fmt = Fmt()
f = fmt


# TODO: move explicit default setting for methods into decorators
say.hr.set(char=u'\u2500', width=40)
say.sep.set(char='-', width=2)
say.title.set(char=u'\u2500', width=15)
say.sep.set(vsep=(1,0))
say.title.set(vsep=1)

# Python 2 might have trouble storing these in instancemethods?

### Helpers

def caller_fmt(*args, **kwargs):
    """
    Like ``fmt()``, but iterpolating strings not from the caller's context, but
    the caller's caller's context. It sounds uber meta, but it helps easily make
    other routines be able to do what ``fmt()`` can do.
    """
    kwargs.setdefault('_callframe', inspect.currentframe().f_back.f_back)
    return fmt(*args, **kwargs)


class FmtException(Exception):

    """
    An exception class that formats its arguments in the calling context.
    Also, an example of how ``caller_fmt`` can be used.
    """
    def __init__(self, message):
        formatted = caller_fmt(message)
        super(FmtException, self).__init__(formatted)
