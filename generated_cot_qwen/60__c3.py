def sum_to_n(n: int):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

# Test cases
assert sum_to_n(30) == 465
assert sum_to_n(100) == 5050
assert sum_to_n(5) == 15
assert sum_to_n(10) == 55
assert sum_to_n(1) == 1
