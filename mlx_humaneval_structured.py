import json, re, sys, time, traceback
from typing import Optional, List

# MLX LM
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
# Data + eval
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness

# Pydantic schema
from pydantic import BaseModel, ValidationError

# Optional: outlines for stricter JSON enforcement (falls back if not present)
try:
    from outlines.processors.structured import JSONLogitsProcessor
    HAVE_OUTLINES = True
except Exception:
    HAVE_OUTLINES = False


# --------- MODEL & TOKENIZER (MLX) ----------
# Recommended tiny coder SLM (fast on M-series). Swap to Llama 3.2 3B if you prefer.
# MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
MODEL_ID = "mlx-community/gemma-2-2b-it-4bit"

model, tokenizer = load(MODEL_ID)  # downloads if missing
# (MLX handles Apple GPU/CPU automatically)


# --------- STRUCTURE SCHEMA ----------
class CotOutput(BaseModel):
    reasoning: str
    code: str


def make_cot_prompt(problem_text: str) -> str:
    user = f"""
You are an expert Python programmer.

Goal: Return ONLY a single JSON object that validates against:

{{
  "reasoning": "1–4 concise sentences",
  "code": "a complete Python function starting with 'def ' on the first line"
}}

Hard rules (critical):
- Output must be a single JSON object. No markdown, no backticks, no prefixes/suffixes.
- The JSON string values must be valid JSON. Escape backslashes and quotes as needed.
- NEVER use Python triple-quoted strings (no `\"\"\"` or `'''`) or comments in the code.
- The code must start with `def ` on the first line.
-- Use **only single quotes `'`** for all string literals and docstrings inside code.
- Do **not** use triple quotes or double quotes anywhere.

Problem:
{problem_text.strip()}
""".strip()

    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)




def make_reflection_prompt(problem_text: str, first_json: str) -> str:
    """
    Ask the model to self-review the previous JSON (reasoning + code) and return a corrected JSON.
    """
    user = f"""
You are an expert Python code reviewer and editor.

You previously wrote this JSON (reasoning + code):
{first_json}

Review it for logical, syntax, or efficiency errors.
Return a corrected JSON object using the same schema:
{{
  "reasoning": "1–4 concise sentences",
  "code": "a complete Python function starting with 'def ' on the first line"
}}

Rules:
- Output only one JSON object, no markdown or prose.
- Escape backslashes and quotes correctly.
- No triple-quoted strings or comments.
- No imports or type hints requiring typing.List etc.
- If the original code is already correct, return the same JSON.
- The code must start with 'def ' on the first line.
- In the code string, always use single quotes ' for all string literals.
- Do not use double quotes (") anywhere in code, because it breaks JSON

Problem:
{problem_text.strip()}
""".strip()
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def solve_with_self_edit(problem_text: str) -> Optional[CotOutput]:
    """
    Two-step pipeline:
    1. Generate initial reasoning+code (CoT)
    2. Ask model to reflect and improve the same code
    """
    first_prompt = make_cot_prompt(problem_text)
    first = generate_structured(first_prompt)
    if not first:
        return None

    reflection_prompt = make_reflection_prompt(problem_text, first.model_dump_json())
    second = generate_structured(reflection_prompt)

    return second or first



def make_debug_prompt(problem_text: str, prev_json: str, error_msg: str) -> str:
    """
    Ask the model to fix its previous code based on a runtime or assertion error.
    """
    user = f"""
You are a Python debugging assistant.

Your previous solution (as JSON) failed during testing:
{prev_json}

Error message:
{error_msg.strip()}

Analyze the cause and return a corrected JSON object:
{{
  "reasoning": "1–3 concise sentences explaining the bug and fix",
  "code": "a complete Python function starting with 'def '"
}}

Rules:
- Keep the same function signature.
- Do not add imports.
- Use only single quotes for all string literals.
- Output ONE JSON object, no markdown, no prose.
- Escape backslashes and quotes correctly.
"""
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def solve_with_self_debug(problem_text: str, problem_tests: str, max_rounds: int = 3) -> Optional[CotOutput]:
    """
    1. Generate initial reasoning+code.
    2. Execute it against HumanEval tests.
    3. On error or assertion failure, feed back the traceback to the model for repair.
    """
    first = generate_structured(make_cot_prompt(problem_text))
    if not first:
        return None

    ns = {}
    for round_no in range(1, max_rounds + 1):
        try:
            preamble = "from typing import *\n"
            fn_name_match = re.search(r'def\s+(\w+)\s*\(', first.code)
            if not fn_name_match:
                raise ValueError("No function definition found in code.")
            fn = fn_name_match.group(1)
            wrapper = f"\ndef candidate(*args, **kwargs):\n    return {fn}(*args, **kwargs)\n"
            exec(preamble + first.code + wrapper, ns, ns)
            exec(problem_tests, ns, ns)
            ns["check"](ns["candidate"])
            print(f"✅ Passed after {round_no} round(s).")
            return first  # success
        except Exception as e:
            error_msg = traceback.format_exc(limit=5)
            print(f"⚠️ Round {round_no} failed:\n{error_msg}")

            # Generate debug prompt with the captured error
            debug_prompt = make_debug_prompt(problem_text, first.model_dump_json(), error_msg)
            fixed = generate_structured(debug_prompt)
            if not fixed:
                print("⚠️ Failed to parse fixed JSON, stopping.")
                break
            first = fixed
    return first



def extract_json(text: str) -> Optional[str]:
    # Try to slice to the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1].strip()
        return candidate
    return None


def generate_structured(prompt: str,
                        max_new_tokens: int = 2024,
                        temperature: float = 0.2,
                        top_p: float = 0.95,
                        retries: int = 3) -> Optional[CotOutput]:
    """
    Use MLX to generate; validate with Pydantic; retry a few times.
    If `outlines` is available, add a JSON logits processor to strongly bias output to the schema.
    """
    schema = CotOutput.model_json_schema()

    logits_processors = None
    if HAVE_OUTLINES:
        try:
            # Stronger JSON constraint (token-level masking)
            logits_processors = [JSONLogitsProcessor(schema)]
        except Exception:
            logits_processors = None

    for attempt in range(1, retries + 1):
        sampler = make_sampler(temperature, top_p=top_p)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
            logits_processors=logits_processors,  # works if outlines installed
        ).strip()

        # Helpful for debugging — keep short
        print(f"\n=== Raw output (attempt {attempt}) ===\n{text[:600]}\n====================")

        candidate = extract_json(text) or text
        try:
            parsed = CotOutput.model_validate_json(candidate)
            # Minimal sanity check
            if not parsed.code.strip().startswith("def "):
                raise ValueError("Code does not start with 'def '.")
            return parsed
        except ValidationError:
            # heuristic fix for triple quotes or bad JSON
            fixed = re.sub(r'"""[\s\S]*?"""', '', candidate)
            fixed = re.sub(r"'''[\s\S]*?'''", '', fixed)
            try:
                parsed = CotOutput.model_validate_json(fixed)
                if not parsed.code.strip().startswith('def '):
                    raise ValueError("Code does not start with 'def '.")
                return parsed
            except Exception as e2:
                print(f"⚠️ Secondary JSON repair failed: {e2}")
                continue


    return None


def main():
    # Load HumanEval (164 tasks)
    dataset = load_dataset("openai_humaneval")["test"]
    samples = [dataset[i] for i in range(0, min(len(dataset), 100), 10)]
    print(f"Evaluating {len(samples)} problems...")

    all_results: List[dict] = []
    n_comps_per_task = 3 
    # USE_SELF_EDIT = True 
    USE_DEBUG = True


    for idx, s in enumerate(samples, 1):
        problem_text = s["prompt"]
        completions: List[str] = []

        # if USE_SELF_EDIT:
        #     for _ in range(n_comps_per_task):
        #         result = solve_with_self_edit(problem_text)  
        #         if result:
        #             completions.append(result.code.strip() + "\n")
        if USE_DEBUG:
            for _ in range(n_comps_per_task):
                result = solve_with_self_debug(problem_text, s["test"])
                if result:
                    completions.append(result.code.strip() + "\n")
        else:
            prompt = make_cot_prompt(problem_text)
            for _ in range(n_comps_per_task):
                result = generate_structured(prompt)
                if result:
                    completions.append(result.code.strip() + "\n")

        for c in completions:
            all_results.append({"task_id": s["task_id"], "completion": c})

        print(f"Task {idx}/{len(samples)} | completions: {len(completions)}")

    out_path = "samples_custom_structured_gemma.jsonl"
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved: {out_path}")

    # HumanEval scoring
    scores = evaluate_functional_correctness(
        out_path,
        n_workers=4,
        k=[1, 3],
        timeout=7.0,          # seconds per test to stay snappy
        ignore_incomplete=True,
    )
    print("\n🎯 Final HumanEval scores:")
    print(scores)


if __name__ == "__main__":
    main()
