"""
The MIT License

Copyright (c) 2017-2020 Mark Douthwaite
"""

import re


def deformat(s: str) -> str:
    """Strip some standard formatting from string."""

    s = str(s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\t", " ", s)
    return s
