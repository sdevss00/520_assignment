def next_smallest(lst):
    if not lst or len(lst) == 1:
        return None
    return sorted(lst)[1]
