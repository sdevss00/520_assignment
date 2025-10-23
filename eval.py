from human_eval.evaluation import evaluate_functional_correctness

if __name__ == "__main__":       
    results = evaluate_functional_correctness(
        "samples_edit_structured_qwen.jsonl",
        n_workers=4,
        k=[1, 3],
        ignore_incomplete=True,
    )
    print(results)