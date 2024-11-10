import json, re, sys
from pathlib import Path
from datasets import load_dataset

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/ssethi/Documents/cot/results/samples_custom_structured_qwen.jsonl"
OUT_DIR = Path("generated_cot_qwen")
OUT_DIR.mkdir(exist_ok=True)

dataset = load_dataset("openai_humaneval")["test"]
tests_by_id = {ex["task_id"]: ex["test"] for ex in dataset}

by_task = {}
with open(IN_PATH) as f:
    for line in f:
        rec = json.loads(line)
        tid = rec["task_id"]
        by_task.setdefault(tid, []).append(rec["completion"])

manifest = []
for tid, completions in by_task.items():
    tid = tid.split('/')[1]
    for i, code in enumerate(completions, 1):
        if not re.search(r"^\s*def\s+\w+\s*\(", code):
            continue
        mod_path = OUT_DIR / f"{tid}__c{i}.py"
        mod_path.write_text(code, encoding="utf-8")
        manifest.append({"task_id": tid, "module": str(mod_path), "index": i})

Path("generated_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {len(manifest)} generated modules into {OUT_DIR}/ and generated_manifest.json")
