from typing import List, Tuple
def find_closest_elements(numbers: List[float]) -> Tuple[float, float] | None:
    if len(numbers) <= 2:
        return None
    return tuple(sorted(numbers)[-2:])
