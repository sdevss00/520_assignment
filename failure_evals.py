import re, json, sys, traceback
from human_eval.data import read_problems

TASK = "HumanEval/20" 


problems = read_problems()
problem = problems[TASK] 


results_path = sys.argv[1]
code = None
with open(results_path) as f:
    for line in f:
        obj = json.loads(line)
        if obj["task_id"] == TASK and obj.get("completion"):
            code = obj["completion"]
            break

if code is None:
    raise SystemExit(f"No completion found for {TASK}")


def_name_match = re.search(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', code, flags=re.MULTILINE)
if def_name_match and def_name_match.group(1) != problem["entry_point"]:
    code = re.sub(
        rf'^(\s*def\s+){re.escape(def_name_match.group(1))}(\s*\()',
        rf'\1{problem["entry_point"]}\2',
        code, count=1, flags=re.MULTILINE
    )


ns = {}
try:
    fn = problem["entry_point"]
    wrapper = f"\n\ndef candidate(*args, **kwargs):\n    return {fn}(*args, **kwargs)\n"

    preamble = "from typing import *\n"
    exec(preamble + code + wrapper, ns, ns)
except Exception as e:
    print("❌ Syntax/runtime error while defining your function:\n", e)
    raise

try:
    exec(problem["test"], ns, ns)
    ns["check"](ns["candidate"])
    print("✅ All asserts passed locally.")

except AssertionError:
    tb = traceback.format_exc()
    print("❌ Assertion failed!\n")
    print(tb)

    # Get all assertion lines from test
    lines = problem["test"].splitlines()
    assert_lines = [l for l in lines if "assert " in l]

    print("\n🔹 All test cases for this task:\n")
    for i, line in enumerate(assert_lines, start=1):
        print(f"{i:2d}: {line.strip()}")

    # Try to isolate the failing assert by running them one by one
    print("\n🔍 Testing assertions individually to find the failing one...\n")
    exec(problem["test"], ns, ns)
    check_src = problem["test"].split("def check(")[-1]
    for i, line in enumerate(assert_lines, start=1):
        local_ns = dict(ns)
        try:
            exec(line.strip(), local_ns, local_ns)
        except AssertionError:
            print(f"❌ Failed assertion #{i}: {line.strip()}")
            break
    else:
        print("⚠️ Could not isolate the exact failing assert.")

except Exception:
    tb = traceback.format_exc()
    print("❌ Exception raised during tests:\n", tb)