import os, json, argparse
from rag import answer_question

def run_eval(namespace: str, qa_path: str):
    golds = []
    preds = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            q = obj["question"]
            gold = obj.get("answer","")
            ans, ctx = answer_question(q, namespace=namespace)
            print("\nQ:", q)
            print("A:", ans)
            print("----")
            preds.append(ans)
            golds.append(gold)
    # naive correctness: presence of any gold keyword in pred
    correct = 0
    total = len(golds)
    for g, p in zip(golds, preds):
        if not g:
            continue
        ok = any(tok.lower() in p.lower() for tok in g.split()[:4])
        correct += int(ok)
    print(f"\nApprox. accuracy (very rough): {correct}/{total} = {correct/total if total else 0:.2%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--questions", default="./sample_questions.jsonl")
    args = ap.parse_args()
    run_eval(args.namespace, args.questions)
