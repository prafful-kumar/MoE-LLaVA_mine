"""
Generate Excel files from eval_results JSON files.
Run this script whenever new results are available to refresh the Excel files.

Usage: python3 excel_results/generate_excel.py
"""
import json
import os
import pandas as pd

RESULTS_DIR = "eval_results"
OUT_DIR = "excel_results"
STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# ── 1. SQA checkpoints (Qwen) ─────────────────────────────────────────────────
def make_sqa_qwen():
    rows = []
    for variant in ["author", "student", "teacher_student"]:
        for step in STEPS:
            f = os.path.join(RESULTS_DIR, "sqa_checkpoints", f"qwen_{variant}_step{step}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            rows.append({
                "variant":      variant,
                "step":         step,
                "accuracy":     d.get("accuracy"),
                "img_accuracy": d.get("img_accuracy"),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "checkpoint_sqa_qwen.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        for variant in df["variant"].unique():
            df[df["variant"] == variant].to_excel(writer, sheet_name=variant, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")
    return df

# ── 2. MME checkpoints (Qwen) ─────────────────────────────────────────────────
def make_mme_qwen():
    rows = []
    for variant in ["author", "student", "teacher_student"]:
        for step in STEPS:
            f = os.path.join(RESULTS_DIR, "mme_checkpoints", f"qwen_{variant}_step{step}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            rows.append({
                "variant":    variant,
                "step":       step,
                "perception": d.get("perception"),
                "cognition":  d.get("cognition"),
                "total":      d.get("total"),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "checkpoint_mme_qwen.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        for variant in df["variant"].unique():
            df[df["variant"] == variant].to_excel(writer, sheet_name=variant, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")
    return df

# ── 3. GQA checkpoints (Qwen) ─────────────────────────────────────────────────
def make_gqa_qwen():
    rows = []
    for variant in ["author", "student", "teacher_student"]:
        for step in STEPS:
            f = os.path.join(RESULTS_DIR, "gqa_checkpoints", f"qwen_{variant}_step{step}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            rows.append({
                "variant":  variant,
                "step":     step,
                "accuracy": d.get("accuracy"),
            })
    if not rows:
        print("GQA checkpoints: no data yet (still running?)")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "checkpoint_gqa_qwen.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        for variant in df["variant"].unique():
            df[df["variant"] == variant].to_excel(writer, sheet_name=variant, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")
    return df

# ── 4. Final results — Qwen ───────────────────────────────────────────────────
def make_final_qwen():
    PAPER = {
        "VQAv2": 76.2, "GQA": 61.5, "VisWiz": 32.6, "SQA": 63.1,
        "TextVQA": 48.0, "POPE_pop_f1": 87.7, "POPE_adv_f1": 85.4,
        "MME_perception": 1291.6, "MMBench": 59.7,
    }
    rows = []

    def load_variant(label, folder):
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
            row["SQA"] = r.get("accuracy")
            row["SQA_img"] = r.get("img_accuracy")

        f = os.path.join(base, "mme.json")
        if os.path.exists(f):
            r = json.load(open(f)).get("results", {})
            row["MME_perception"] = r.get("perception", {}).get("total") if isinstance(r.get("perception"), dict) else r.get("perception")
            row["MME_cognition"]  = r.get("cognition",  {}).get("total") if isinstance(r.get("cognition"),  dict) else r.get("cognition")
            row["MME_total"]      = r.get("total")

        f = os.path.join(base, "pope.json")
        if os.path.exists(f):
            r = json.load(open(f)).get("results", {})
            for cat in ["popular", "adversarial", "random"]:
                c = r.get(cat, {})
                row[f"POPE_{cat}_acc"] = round(c.get("accuracy", 0) * 100, 2) if c.get("accuracy", 0) <= 1 else c.get("accuracy")
                row[f"POPE_{cat}_f1"]  = round(c.get("f1", 0) * 100, 2)       if c.get("f1", 0) <= 1       else c.get("f1")

        return row

    rows.append(load_variant("author",          "qwen_author"))
    rows.append(load_variant("student",         "qwen_student"))
    rows.append(load_variant("teacher_student", "qwen_TS"))

    # Paper reference row
    paper_row = {"variant": "paper_reference"}
    paper_row.update(PAPER)
    rows.append(paper_row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "final_results_qwen.xlsx")
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Saved: {path}  ({len(df)} rows)")
    return df

# ── 5. Final results — Phi2 ───────────────────────────────────────────────────
def make_final_phi2():
    PAPER = {
        "VQAv2": 77.6, "GQA": 61.4, "VisWiz": 43.9, "SQA": 68.5,
        "TextVQA": 51.4, "POPE_pop_f1": 86.4, "POPE_adv_f1": 84.9,
        "MME_perception": 1423.0, "MMBench": 65.2,
    }
    rows = []

    def load_phi2(label, folder):
        row = {"variant": label}
        base = os.path.join(RESULTS_DIR, folder)

        f = os.path.join(base, "gqa.json")
        if os.path.exists(f):
            d = json.load(open(f))
            row["GQA"] = d.get("results", {}).get("accuracy") or d.get("accuracy")

        f = os.path.join(base, "textvqa.json")
        if os.path.exists(f):
            d = json.load(open(f))
            row["TextVQA"] = d.get("results", {}).get("accuracy") or d.get("accuracy")

        f = os.path.join(base, "sqa.json")
        if os.path.exists(f):
            d = json.load(open(f))
            r = d.get("results", d)
            row["SQA"]     = r.get("accuracy")
            row["SQA_img"] = r.get("img_accuracy")

        f = os.path.join(base, "mme.json")
        if os.path.exists(f):
            d = json.load(open(f))
            r = d.get("results", d)
            perc = r.get("perception", {})
            cog  = r.get("cognition", {})
            row["MME_perception"] = perc.get("total") if isinstance(perc, dict) else perc
            row["MME_cognition"]  = cog.get("total")  if isinstance(cog,  dict) else cog
            row["MME_total"]      = r.get("total")

        f = os.path.join(base, "pope.json")
        if os.path.exists(f):
            d = json.load(open(f))
            r = d.get("results", d)
            for cat in ["popular", "adversarial", "random"]:
                c = r.get(cat, {})
                row[f"POPE_{cat}_acc"] = round(c.get("accuracy", 0) * 100, 2) if c.get("accuracy", 0) <= 1 else c.get("accuracy")
                row[f"POPE_{cat}_f1"]  = round(c.get("f1", 0) * 100, 2)       if c.get("f1", 0) <= 1       else c.get("f1")

        return row

    rows.append(load_phi2("author",  "phi2_author"))
    rows.append(load_phi2("student", "phi2_student_final"))

    paper_row = {"variant": "paper_reference"}
    paper_row.update(PAPER)
    rows.append(paper_row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "final_results_phi2.xlsx")
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Saved: {path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_sqa_qwen()
    make_mme_qwen()
    make_gqa_qwen()
    make_final_qwen()
    make_final_phi2()
    print("\nAll Excel files generated in excel_results/")
