def strange_sort_list(lst):
    if not lst: return []
    min_val = min(lst)
    max_val = max(lst)
    if min_val == max_val:
        return [min_val] * len(lst)
    return [min_val] + strange_sort_list([x for x in lst if x != min_val or x != max_val]) + [max_val]
