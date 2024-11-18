
_memento_cache = {}

def memento_factory(name, func):
    """
    Return a memoizing metaclass with the given name and key function.
    That makes this a parametrized meta-metaclass, which is probably
    the most meta thing you've ever seen. If it isn't, both congratulations
    and sympathies are in order!
    """

    def call(cls, *args, **kwargs):
        key = func(cls, args, kwargs)
        try:
            return _memento_cache[key]
        except KeyError:
            instance = type.__call__(cls, *args, **kwargs)
            _memento_cache[key] = instance
            return instance

    mc = type(name, (type,), {'__call__': call})
    return mc


MementoMetaclass = memento_factory("MementoMetaclass",
                                   lambda cls, args, kwargs: (cls, ) + args + tuple(kwargs.items()))


def with_metaclass(meta, base=object):
    """
    Create a base class with a metaclass. Compatible across Python 2 and Python
    3. Extension of the with_metaclass() found in the six module.
    """

    basetuple = base if isinstance(base, tuple) else (base,)
    return meta("NewBase", basetuple, {})


# even simpler front-end
mementos = with_metaclass(MementoMetaclass, object)

# Some reading:
# http://bytes.com/topic/python/answers/40084-parameterized-metaclass-metametaclass
# http://www.acooke.org/cute/PythonMeta0.html
# http://www.python.org/dev/peps/pep-3115/
