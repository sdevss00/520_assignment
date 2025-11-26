import importlib, pytest
mod = importlib.import_module("generated_cot_qwen.20__c3") 
find_closest_elements = getattr(mod, "find_closest_elements") 

def test_condition_true_branch():
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (3.9, 4.0)


def test_empty_input_exit_path():
    assert find_closest_elements([]) == ()


def test_minimal_input():
    assert find_closest_elements([1.0]) == ()


def test_maximal_input():
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2, 3.1, 4.1, 5.1]) == (2.2, 3.1)


def test_edge_case_negative_numbers():
    assert find_closest_elements([-1.0, -2.0, -3.0, -4.0, -5.0, -2.2]) == (-3.0, -2.2)


def test_edge_case_large_numbers():
    assert find_closest_elements([100.0, 200.0, 300.0, 400.0, 500.0, 2.2]) == (200.0, 2.2)


def test_invalid_type_input_fails():
    with pytest.raises(AssertionError):
        find_closest_elements("not a list")

def test_minimal_valid_two_numbers():
    assert find_closest_elements([1.0, 2.0]) == (1.0, 2.0)

def test_duplicate_values_valid_closest_pair():
    assert find_closest_elements([1.0, 1.0, 2.0]) == (1.0, 2.0)

def test_unsorted_list_valid():
    assert find_closest_elements([2.0, 1.0, 3.0]) == (1.0, 2.0)

def test_length_less_than_two_fails():
    with pytest.raises(AssertionError):
        find_closest_elements([1.0])

def test_negative_floats_allowed():
    assert find_closest_elements([-1.0, 2.0]) == (-1.0, 2.0)