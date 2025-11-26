from mlx_lm import load, generate

import importlib.util, json, re, types
from pathlib import Path
import pytest
from datasets import load_dataset

# MODEL_ID = "mlx-community/gemma-2-2b-it-4bit"
MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

model, tokenizer = load(MODEL_ID)







def make_prompt(assertions, existing_tests):
    prompt = f'''
You are an expert Python software tester specializing in coverage-driven test generation.

Your job is to generate **new Pytest unit tests** for a target function, based entirely on the function’s **formal specification**, which is provided as a set of `assert` statements.

You MUST use the assertions as the sole source of truth for:
- valid vs invalid inputs,
- expected outputs,
- failure conditions,
- boundary behaviors.

============================================================
### INPUT YOU WILL RECEIVE
The following arguments will be supplied to you:

1. **function_signature**  
   Example:  
   `def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:`

2. **assertions**  
   A block of Python `assert` statements which define the complete desired behavior.  
   These asserts describe:
   - input preconditions,  
   - output postconditions,  
   - ordering, membership, and correctness constraints.

3. **existing_tests**  
   A block of already existing Pytest tests.  
   You MUST generate **new** tests, not duplicates.

============================================================
### YOUR TASK
Create **only new Pytest-style test functions**, each of which:

- Covers a previously untested branch, edge case, or assertion outcome.
- Exercises both:
  - valid inputs that must satisfy all assertions, and
  - invalid inputs that must raise `AssertionError`.
- Reflects the exact behavior defined by the assertion block.
- Does NOT depend on any internal implementation (the assertions are the oracle).
- Has a unique descriptive test name such as:
  `test_invalid_type_input_fails`,  
  `test_minimal_valid_two_numbers`,  
  `test_duplicate_values_valid_closest_pair`,  
  `test_unsorted_list_valid`,  
  `test_length_less_than_two_fails`,  
  `test_negative_floats_allowed`,  
  etc.

============================================================
### STRICT OUTPUT RULES
- Output **only** the new test function definitions.
- No text, no comments, no markdown fences, no explanation.
- No print statements.
- Use only `assert` and `with pytest.raises(AssertionError):`.
- Do not rewrite or include the assertions or function signature in the output.
- Do not duplicate existing tests (provided via `existing_tests`).

============================================================
### DATA PROVIDED TO YOU
Assertions:
{assertions}

Existing Tests:
{existing_tests}

============================================================
Provide ONLY the new Pytest test function definitions and NOTHING ELSE.



    '''.strip()

    prompt_2 = '''

You are an expert Python software tester specializing in coverage-driven test generation.

Your job is to generate **new Pytest unit tests** for a target function, based entirely on the function’s **formal specification**, which is provided as a list of Python `assert` statements.

IMPORTANT (READ CAREFULLY):
- The assertions are the **sole source of truth** for the function’s behavior.
- You MUST derive expected outputs **only** by logically applying the assertions.
- You MUST NOT use or rely on the function implementation, patterns from previous problems, or assumptions about lists, numbers, tuples, or numeric ordering.
- You MUST enforce the exact semantics that the assertions describe (e.g., “result must start with the input string”, “result must be a palindrome”, “result must be the *shortest possible* palindrome beginning with the input”).
- When computing expected outputs, you MUST carry out the correct algorithm implied by the assertions (e.g., for palindromes: find the longest palindromic suffix, then append the reversed remaining prefix).

============================================================
### INPUTS THAT WILL BE PROVIDED TO YOU

1. **function_signature**  
   Example:  
   `def make_palindrome(string: str) -> str:`

2. **assertions**  
   A complete block of Python `assert` statements defining:
   - allowed input types,
   - required output types,
   - structural constraints (palindrome, startswith),
   - correctness constraints (shortest possible extension).

3. **existing_tests**  
   A set of Pytest tests already written.
   You MUST generate new tests that are not duplicates.

============================================================
### YOUR TASK

Generate **new Pytest unit tests**, each of which:

- Covers a previously untested assertion outcome or edge case.
- Includes:
  - valid input cases that must satisfy *all* assertions,
  - invalid inputs that must raise AssertionError (only when required by the assertions),
  - boundary cases (empty string, single char, repeated chars),
  - cases involving overlapping palindromic suffixes,
  - cases where the shortest-palindrome property matters.

- For each valid test case:
  You MUST compute the expected output EXACTLY according to the specification —  
  not the implementation, not heuristics, and not guesses.  
  In particular:
  - The result must start with the input.
  - The result must be a palindrome.
  - The result must be the shortest such palindrome.
  - If the input is already a palindrome, the output is identical.

Examples of **valid naming style**:
  `test_empty_string_returns_empty`,
  `test_single_character_is_palindrome`,
  `test_shortest_extension_abc`,
  `test_overlapping_suffix_abab`,
  `test_already_palindrome_returns_self`,
  `test_invalid_nonstring_input_fails`.

============================================================
### STRICT OUTPUT RULES

-When generating expected outputs for valid inputs, you MUST compute the minimal palindrome according to the assertions using this exact constructive rule:

1. Identify the longest **palindromic suffix** of the input string.
2. Let prefix = the part of the string before that suffix.
3. The result must be:  input_string + reverse(prefix).
You MUST apply this exact algorithm for every test case.
- Output ONLY Python test function definitions.
- No comments, no explanations, no markdown fences.
- No print statements.
- Do not rewrite or include the assertions or signature.
- Do not duplicate any test in `existing_tests`.

============================================================
### DATA PROVIDED TO YOU

Assertions:
{assertions}

Existing Tests:
{existing_tests}

============================================================
Provide ONLY the new Pytest test function definitions and NOTHING ELSE.
    '''.strip()
    messages = [{"role": "user", "content": prompt_2}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



dataset = load_dataset("openai_humaneval")["test"]
TESTS = {ex["task_id"]: ex["test"] for ex in dataset}
PROMPTS = {ex["task_id"]: ex["prompt"] for ex in dataset}

print(PROMPTS['HumanEval/10'])


assertions_correct = {
    
    "HumanEval/20":"""
assert isinstance(numbers, list)
assert len(numbers) >= 2
assert all(isinstance(x, float) for x in numbers)
assert isinstance(result, tuple)
assert len(result) == 2
assert all(isinstance(x, float) for x in result)
assert result[0] <= result[1]
assert result[0] in numbers and result[1] in numbers
assert all(
    abs(result[1] - result[0]) <= abs(numbers[j] - numbers[i])
    for i in range(len(numbers))
    for j in range(len(numbers))
    if i != j
)
""".strip(),

"HumanEval/10":"""

assert isinstance(numbers, list)
assert len(numbers) >= 2
assert all(isinstance(x, float) for x in numbers)

assert isinstance(result, tuple)
assert len(result) == 2
assert all(isinstance(x, float) for x in result)

assert result[0] <= result[1]

assert result[0] in numbers and result[1] in numbers

assert all(
    abs(result[1] - result[0]) <= abs(numbers[j] - numbers[i])
    for i in range(len(numbers))
    for j in range(len(numbers))
    if i != j
)


"""


}

existing_test = [TESTS['HumanEval/20'], TESTS['HumanEval/10']]
prompt = make_prompt(assertions_correct["HumanEval/10"], existing_test[1])


tests = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
        ).strip()


print(tests)