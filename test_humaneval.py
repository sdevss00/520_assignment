import importlib.util, json, re, types
from pathlib import Path
import pytest
from datasets import load_dataset

MANIFEST = json.loads(Path("generated_manifest_qwen.json").read_text())
dataset = load_dataset("openai_humaneval")["test"]
TESTS = {ex["task_id"]: ex["test"] for ex in dataset}

def load_module_from_path(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def find_first_function_name(src_text: str) -> str:
    m = re.search(r"^\s*def\s+(\w+)\s*\(", src_text, flags=re.M)
    assert m, "No function definition found"
    return m.group(1)

CASES = []
for rec in MANIFEST:
    p = Path(rec["module"])
    src = p.read_text(encoding="utf-8")
    fn = find_first_function_name(src)
    CASES.append(pytest.param(rec["task_id"], str(p), fn, id=f"{rec['task_id']}__c{rec['index']}"))

@pytest.mark.parametrize("task_id, module_path, fn_name", CASES)
def test_humaneval_candidate(task_id, module_path, fn_name):
    mod = load_module_from_path(module_path)
    assert hasattr(mod, fn_name)
    target_fn = getattr(mod, fn_name)

    # build candidate wrapper that HumanEval tests expect
    ns = {"candidate": lambda *a, **kw: target_fn(*a, **kw)}
    if task_id not in TESTS:
        for k in TESTS.keys():
            if f"/{task_id}_" in k or k.endswith(f"/{task_id}"):
                task_id = k
                break
    exec(TESTS[task_id], ns, ns)
    # run the check
    ns["check"](ns["candidate"])
