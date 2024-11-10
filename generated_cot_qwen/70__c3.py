def strange_sort_list(lst):
    if not lst:
        return []
    result = []
    while lst:
        min_val = min(lst)
        max_val = max(lst)
        index_min = lst.index(min_val)
        index_max = lst.index(max_val)
        result.extend([min_val, max_val])
        lst.pop(index_min)
        lst.pop(index_max)
    return result
