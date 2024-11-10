def is_happy(s):
    if len(s) < 3:
        return False
    last_two = set()
    for i in range(len(s) - 2):
        if s[i] in last_two or s[i+1] in last_two or s[i+2] in last_two:
            return False
        last_two.add(s[i])
    return True
