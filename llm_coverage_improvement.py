import importlib.util, json, re, types
from pathlib import Path
import pytest
from datasets import load_dataset

from mlx_lm import load, generate



def find_first_function_name(src_text: str) -> str:
    m = re.search(r"^\s*def\s+(\w+)\s*\(", src_text, flags=re.M)
    assert m, "No function definition found"
    return m.group(1)




def make_prompt(func, tests):
    prompt = f'''
    
You are an expert Python software tester specializing in coverage-driven test generation.

Goal:
Generate **only new** Pytest-style unit tests for the following Python function.
These tests must specifically increase **branch coverage** by triggering:
- all remaining untested conditions, else-paths, or early exits,
- both True and False outcomes of conditionals,
- edge inputs (empty, None, minimal, maximal, etc.).

Constraints:
- The function code is provided below.
- Do not modify or rewrite the function.
- Do not repeat existing tests; only add new ones.
- Each test must have a **unique, descriptive name** like:
  `test_condition_true_branch`, `test_empty_input_exit_path`, etc.
- Each test must be fully runnable with `pytest` and assert actual outcomes.
- Avoid any print statements or comments.
- Your output must contain **only the test function definitions**, nothing else.
- Do **not** include explanations, summaries, or markdown fences.

- PROVIDE ONLY THE TEST CASES NO OTHER TEXTS

Function under test:
{func}

Existing tests (HumanEval baseline):
{tests.strip()}


- PROVIDE ONLY THE TEST CASES NO OTHER TEXTS

    '''.strip()

    prompt_pal = f'''
You are an expert Python software tester specializing in coverage-driven test generation.

Goal:
Generate **only new** Pytest-style unit tests for the following two functions from HumanEval/20 (variants c2 and c3).
These tests must specifically increase **branch coverage** by triggering:
- all remaining untested conditions, else-paths, or early exits,
- both True and False outcomes of conditional checks,
- edge inputs (empty, None, minimal, maximal, duplicates, negative values, etc.),
- and exception or early-return scenarios where applicable.

Constraints:
- The function code is provided below (for c2 and c3 respectively).
- Do not modify or rewrite the functions.
- Do not repeat existing tests; only add new ones.
- Each test must have a **unique, descriptive name** such as:
  `test_c2_nameerror_path`, `test_c2_indexerror_branch`, `test_c3_empty_input_returns_none`, `test_c3_duplicates_sorted`, etc.
- Each test must be fully runnable with `pytest` and assert actual outcomes or exceptions.
- Avoid any print statements, comments, or explanations.
- Your output must contain **only the test function definitions**, nothing else.
- Do **not** include markdown fences or any text outside the test functions.

Coverage focus per function:
- **Function c2:** Trigger all remaining exception and success branches (e.g., missing imports, index errors, and valid return cases).
- **Function c3:** Exercise both conditional outcomes for the input-length check, along with unordered, negative, and duplicate inputs.

Existing tests (HumanEval baseline + LLM iteration 1):
{tests.strip()}

- PROVIDE ONLY THE TEST CASES NO OTHER TEXTS


'''.strip()
    messages = [{"role": "user", "content": prompt_pal}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def finalize_llm_tests(raw_text: str, out_path: str, func_name: str, module_path: str):

    cleaned = re.sub(r"```python|```", "", raw_text).strip()
    blocks, current = [], []

    for line in cleaned.splitlines():
        if line.strip().startswith("def test_"):
            if current:
                blocks.append("\n".join(current))
                current = []
            current.append(line.rstrip())
        elif current:
            current.append(line.rstrip())

    if current:
        blocks.append("\n".join(current))

    blocks = [b for b in blocks if "assert" in b]
    blocks = [b for b in blocks if b.strip().endswith(")") or b.strip().endswith("]")]

    seen, unique_blocks = set(), []
    for b in blocks:
        norm = re.sub(r"\s+", " ", b).strip()
        if norm not in seen:
            seen.add(norm)
            unique_blocks.append(b)

    import_block = f"""import importlib, pytest
        mod = importlib.import_module("{module_path}") {func_name} = getattr(mod, "{func_name}") """

    final_code = import_block + "\n\n".join(unique_blocks) + "\n"

    Path(out_path).write_text(final_code, encoding="utf-8")
    return final_code







MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
model, tokenizer = load(MODEL_ID)



MANIFEST = json.loads(Path("generated_manifest_qwen.json").read_text())
dataset = load_dataset("openai_humaneval")["test"]
TESTS = {ex["task_id"]: ex["test"] for ex in dataset}
PROMPTS = {ex["task_id"]: ex["prompt"] for ex in dataset}




# task_id = ['70', '10']
task_id = ['20']

candidates = ["c1", "c2", "c3"]

for task in task_id:
    if task not in TESTS:
            for k in TESTS.keys():
                if f"/{task}_" in k or k.endswith(f"/{task}"):
                    task = k
                    break
    
    prompt = PROMPTS[task].strip()
    tests = TESTS[task].strip()
    print('------------')
    print(prompt, tests)
    print('------------')

    try:
        func_name = find_first_function_name(prompt)
    except AssertionError:
        func_name = "unknown_function"

    for cand in candidates:
        t_id = task.split('/')[1]
        module_path = f"/Users/ssethi/Documents/cot/generated_cot_qwen/{t_id}__{cand}"
        output_file = f"{t_id}__{cand}_new_tests.py"
        llm_prompt = make_prompt(prompt, tests)
        new_tests = generate(
            model,
            tokenizer,
            prompt=llm_prompt,
            max_tokens=300,
        ).strip()

        code = re.sub(r"^```python|```$", "", new_tests, flags=re.MULTILINE).strip()
        print('==========')
        print(code)
        print('==========')
        finalize_llm_tests(code, output_file, func_name, module_path)
        print(f"Wrote {output_file} with import from {module_path}")

    print("=" * 80 + "\n")









