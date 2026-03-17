"""
Generate pivot-style Excel tables: rows=steps, columns=variants.
Format matches:
    step | Author | Student | TS
    1    | ...    | ...     | ...

Run: python3 eval_results_mine/generate_tables.py
"""
import json
import os
import pandas as pd

RESULTS_DIR = "eval_results"
OUT_DIR = "eval_results_mine"
STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

QWEN_VARIANTS = {
    "author":          "Author",
    "student":         "Student",
    "teacher_student": "TS",
}

def make_pivot(data_dict, steps=STEPS):
    """data_dict: {variant_label: [values in step order]}"""
    df = pd.DataFrame({"step": steps})
    for label, values in data_dict.items():
        df[label] = values
    return df

def load_checkpoint_metric(benchmark, variant, step, key):
    f = os.path.join(RESULTS_DIR, f"{benchmark}_checkpoints",
                     f"qwen_{variant}_step{step}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    # support nested keys like "results.accuracy"
    if "." in key:
        k1, k2 = key.split(".", 1)
        return d.get(k1, {}).get(k2)
    return d.get(key)

def make_qwen_tables():
    benchmarks = {
        "mme_perception": ("mme",  "perception"),
        "mme_cognition":  ("mme",  "cognition"),
        "mme_total":      ("mme",  "total"),
        "sqa_accuracy":   ("sqa",  "accuracy"),
        "sqa_img_accuracy":("sqa", "img_accuracy"),
        "gqa_accuracy":   ("gqa",  "accuracy"),
    }

    path = os.path.join(OUT_DIR, "qwen_checkpoint_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, (bench, metric) in benchmarks.items():
            data = {}
            for var_key, var_label in QWEN_VARIANTS.items():
                values = []
                for step in STEPS:
                    v = load_checkpoint_metric(bench, var_key, step, metric)
                    values.append(round(v, 4) if v is not None else None)
                data[var_label] = values

            df = make_pivot(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet '{sheet_name}': {df.shape}")

    print(f"Saved: {path}")

def make_qwen_final_table():
    """One sheet per benchmark, one row per variant, with all sub-metrics."""
    path = os.path.join(OUT_DIR, "qwen_final_results.xlsx")

    # Define final result folders and their display labels
    variants = {
        "qwen_author": "Author",
        "qwen_student": "Student",
        "qwen_TS":      "TS",
    }

    PAPER = {
        "GQA":            61.5,
        "TextVQA":        48.0,
        "SQA":            63.1,
        "SQA_img":        None,
        "MME_perception": 1291.6,
        "MME_cognition":  None,
        "MME_total":      None,
        "POPE_pop_acc":   88.6,
        "POPE_pop_f1":    87.7,
        "POPE_adv_acc":   86.1,
        "POPE_adv_f1":    85.4,
        "POPE_rand_acc":  88.7,
        "POPE_rand_f1":   88.0,
    }

    rows = []
    for folder, label in variants.items():
        row = {"variant": label}
        base = os.path.join(RESULTS_DIR, folder)

        f = os.path.join(base, "gqa.json")
        if os.path.exists(f):
            row["GQA"] = json.load(open(f)).get("results", {}).get("accuracy")

        f = os.path.join(base, "textvqa.json")
        if os.path.exists(f):
            row["TextVQA"] = json.load(open(f)).get("results", {}).get("accuracy")

        f = os.path.join(base, "sqa.json")
        if os.path.exists(f):
            r = json.load(open(f)).get("results", {})
            row["SQA"]     = r.get("accuracy")
            row["SQA_img"] = r.get("img_accuracy")

        f = os.path.join(base, "mme.json")
        if os.path.exists(f):
            r = json.load(open(f)).get("results", {})
            perc = r.get("perception", {})
            cog  = r.get("cognition",  {})
            row["MME_perception"] = perc.get("total") if isinstance(perc, dict) else perc
            row["MME_cognition"]  = cog.get("total")  if isinstance(cog,  dict) else cog
            row["MME_total"]      = r.get("total")

        f = os.path.join(base, "pope.json")
        if os.path.exists(f):
            r = json.load(open(f)).get("results", {})
            for cat, short in [("popular","pop"), ("adversarial","adv"), ("random","rand")]:
                c = r.get(cat, {})
                acc = c.get("accuracy", 0)
                f1  = c.get("f1", 0)
                row[f"POPE_{short}_acc"] = round(acc * 100, 2) if acc <= 1 else round(acc, 2)
                row[f"POPE_{short}_f1"]  = round(f1  * 100, 2) if f1  <= 1 else round(f1,  2)

        rows.append(row)

    # Add paper reference
    paper_row = {"variant": "Paper (Qwen)"}
    paper_row.update(PAPER)
    rows.append(paper_row)

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Saved: {path}  ({len(df)} rows × {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Qwen checkpoint tables ===")
    make_qwen_tables()
    print("\n=== Qwen final results ===")
    make_qwen_final_table()
    print("\nDone. Files in eval_results_mine/")
