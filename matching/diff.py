#diff.py
#finds discrepancies within similar sentences
def string_diff(str1, str2):
    BOLD = '\033[1m'
    END = '\033[0m'

    one = str1.split(' ')
    two = str2.split(' ')
    out = []
    for x in two:
        if x not in one:
            out.append(x)
    for x in one:
        if x not in two:
            out.append(x)

    one_out = []
    for x in one:
        if x not in out:
            one_out.append(x)
        else:
            one_out.append(BOLD + x + END)
    two_out = []
    for x in two:
        if x not in out:
            two_out.append(x)
        else:
            two_out.append(BOLD + x + END)

    print(' '.join(one_out))
    print(' '.join(two_out))
    return out
