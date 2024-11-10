def next_smallest(lst):
    if not lst or len(lst) == 1:
        return None
    smallest = float('inf')
    second_smallest = float('inf')
    for num in lst:
        if num < smallest:
            second_smallest = smallest
            smallest = num
        elif smallest < num < second_smallest:
            second_smallest = num
    return second_smallest
