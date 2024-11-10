def make_palindrome(string: str) -> str:
    if not string:
        return string
    for i in range(len(string) - 2, -1, -1): 
        if string[i:] == string[i:][::-1]:
            return string[:i] + string[::-1]