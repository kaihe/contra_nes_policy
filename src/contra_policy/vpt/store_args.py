"""``store_args`` — the VPT ``__init__`` decorator, extracted to break a cycle.

Upstream this lives in ``vpt_lib/minecraft_util.py``, whose only other content is the
Minecraft action-head factory (and which drags in ``action_head.py``). Both
``util.py`` and ``masked_attention.py`` need the decorator and ``util`` imports
``masked_attention``, so it gets its own leaf module here.
"""

import functools
import inspect


def store_args(method):
    """Decorator that stores every ``__init__`` argument as an attribute of self."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        args = defaults.copy()
        args.update(zip(arg_names, positional_args[1:]))
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper
