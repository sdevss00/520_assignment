import importlib, pytest
mod = importlib.import_module("generated_cot_qwen.10__c1") 
make_palindrome = getattr(mod, "make_palindrome") 



def test_empty_string_returns_empty():
    assert make_palindrome("") == ""

def test_single_character_is_palindrome():
    assert make_palindrome("a") == "a"

def test_non_palindrome_goes_else_branch():
    assert make_palindrome("abc") == "abcba"

def test_overlapping_suffix_abab():
    assert make_palindrome("abab") == "abababa"

def test_invalid_nonstring_input_fails():
    with pytest.raises(TypeError):
        make_palindrome(123)