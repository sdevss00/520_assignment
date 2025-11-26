from mlx_lm import load, generate

import importlib.util, json, re, types
from pathlib import Path
import pytest
from datasets import load_dataset

# MODEL_ID = "mlx-community/gemma-2-2b-it-4bit"
MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

model, tokenizer = load(MODEL_ID)







def make_prompt(func_sign, nl_desc):
    prompt = f'''
Your task is to write ONLY *formal specifications* in the form of Python `assert`
statements for the following function.

This is NOT code generation.  
This is NOT test-case generation.  
This is NOT implementation.

You MUST write logical constraints ONLY.

===========================================================
### STRICT, NON-NEGOTIABLE RULES
===========================================================

1. **You MUST refer to the function input ONLY as `numbers\string`.**
2. **You MUST refer to the function output ONLY as `result`.**

3. **You MUST write both:**
   - Preconditions on `numbers\string`
   - Postconditions on `result`

4. **You MUST express the definition of "closest pair\make_palindrome":**
   - `result` must be a palindrome.
   - `result` must start with the input `string`.
   - `result` must be the **shortest** palindrome that starts with `string`
     (i.e., no strictly shorter palindrome may start with `string`).


5. **Hard Forbidden:**
   - DO NOT call the function (e.g., no `make_palindrome(...)`).
   - DO NOT write the function implementation.
   - DO NOT write test cases.
   - DO NOT invent constraints not in the description.
   - DO NOT create new variables (no helper variables, no loops, no lambdas).
   - DO NOT mutate data or perform I/O (print, read, write).
   - DO NOT use randomness or timing.

6. **The output MUST be ONLY assert statements**,  
   each describing a logical requirement.

===========================================================
### Function Signature:
{func_sign}

### Natural Language Description:
{nl_desc}
===========================================================

### REQUIRED ASSERTIONS
Your assert output MUST include ALL of the following:

1. That `string` is a `str`.
2. That `result` is a `str`.
3. That `result` starts with `string`.
4. That `result` is a palindrome (`result == result[::-1]`).
5. That `result` is the **shortest** palindrome that starts with `string`.
   Express minimality as:
   - For every prefix of `result` that is strictly shorter than `result`,
     it is NOT both a palindrome and starts with `string`.


===========================================================
### OUTPUT FORMAT (CRITICAL)
- Output ONLY raw Python assert statements.
- No text, no comments, no explanation.
- No code fences.
- No blank lines unless between asserts.

===========================================================
BEGIN OUTPUT NOW:

    '''.strip()
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



dataset = load_dataset("openai_humaneval")["test"]
TESTS = {ex["task_id"]: ex["test"] for ex in dataset}
PROMPTS = {ex["task_id"]: ex["prompt"] for ex in dataset}

print(PROMPTS['HumanEval/10'])


func_sign = ['def find_closest_elements(numbers: List[float]) -> Tuple[float, float]', 'def make_palindrome(string: str) -> str:']
nl_desc = ['From a supplied list of numbers (of length at least two), select and return two numbers that are the closest to each other and return them in order (smaller number, larger number).','Find the shortest palindrome that begins with a supplied string.The algorithm is: - Find the longest suffix of the input string that is already a palindrome. - Then append the reverse of the prefix that comes before this suffix.']


prompt = make_prompt(func_sign[0], nl_desc[1])

assertions = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=300,
        ).strip()


print(assertions)