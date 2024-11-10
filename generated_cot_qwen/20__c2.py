from typing import List, Tuple
def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    numbers.sort()
    left_index = bisect.bisect_left(numbers, numbers[0])
    right_index = bisect.bisect_right(numbers, numbers[-1])
    return numbers[left_index], numbers[right_index]


