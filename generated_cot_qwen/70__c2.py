def strange_sort_list(lst):
    if not lst:
        return []
    result = []
    while lst:
        min_val = min(lst)
        result.append(min_val)
        lst.remove(min_val)
        if lst:
            max_val = max(lst)
            result.append(max_val)
            lst.remove(max_val)
        else:
            break
    return result
