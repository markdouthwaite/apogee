import re


def deformat(s):
    s = str(s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\t", " ", s)
    return s
