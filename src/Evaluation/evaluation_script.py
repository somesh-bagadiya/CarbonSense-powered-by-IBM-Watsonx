import re
import json
import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

# Constants
BERT_PASS_CUTOFF = 0.75
NUMERIC_THRESHOLD_PERCENT = 1
KEYWORD_COVERAGE_CUTOFF = 0.75

# Step 1: Load JSON data
with open("agent_evaluation_results.json", "r") as f:
    eval_data = json.load(f)

# Step 2: Separate by category
numerical_data = [d for d in eval_data if d.get("category") == "numerical_data"]
semantic_data = [d for d in eval_data if d.get("category") == "semantic_explanation_data"]
keyword_data = [d for d in eval_data if d.get("category") == "keywords_data"]

# ------------------ 🔍 SEMANTIC EVALUATION ------------------ #
def evaluate_semantic(data):
    results = []
    refs = [x["expected"] for x in data]
    preds = [x["actual_response"].get("response_text", "") for x in data]

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(refs, preds)]

    _, _, f1_scores = bert_score_fn(preds, refs, lang="en", verbose=False)
    f1_scores = f1_scores.tolist()

    for i, item in enumerate(data):
        status = "Pass" if f1_scores[i] >= BERT_PASS_CUTOFF else "Fail"
        results.append({
            "Prompt": item["prompt"],
            "Expected Response": refs[i],
            "LLM Response": preds[i],
            "ROUGE_L_F1": round(rouge_l_scores[i] * 100, 2),
            "BERTScore_F1": round(f1_scores[i] * 100, 2),
            "Pass/Fail": status
        })
    return pd.DataFrame(results)

# ------------------ 🔢 NUMERICAL EVALUATION ------------------ #
def evaluate_numerical(data):
    results = []
    for item in data:
        prompt = item['prompt']
        expected_raw = item['expected']
        response_text = item['actual_response'].get("response_text", "")

        # Normalize expected value
        expected = expected_raw if isinstance(expected_raw, (int, float)) else (
            expected_raw.get("value") if isinstance(expected_raw, dict) and "value" in expected_raw else None
        )

        match = re.search(r'(\d+\.\d+|\d+)', response_text)
        try:
            predicted = float(match.group()) if match else None
        except:
            predicted = None

        if predicted is not None and expected is not None:
            abs_err = abs(predicted - expected)
            rel_err = (abs_err / expected) * 100 if expected != 0 else 0
            status = "Pass" if rel_err <= NUMERIC_THRESHOLD_PERCENT else "Fail"
        else:
            abs_err = rel_err = None
            status = "Invalid"

        results.append({
            "Prompt": prompt,
            "Expected": expected,
            "LLM_Response": response_text,
            "Metric_1": predicted,
            "Metric_2": abs_err,
            "Metric_3": rel_err,
            "Pass/Fail": status,
            "Evaluation_Type": "Numerical"
        })

    return pd.DataFrame(results)

# ------------------ 🔑 KEYWORD EVALUATION ------------------ #
def evaluate_keywords(data):
    results = []

    for row in data:
        if "expected" not in row or "actual_response" not in row:
            continue

        prompt = row["prompt"]
        keywords = row["expected"]
        response_text = row["actual_response"].get("response_text", "")

        matched = [kw for kw in keywords if kw.lower() in response_text.lower()]
        coverage = len(matched) / len(keywords)
        status = "Pass" if coverage >= KEYWORD_COVERAGE_CUTOFF else "Fail"

        results.append({
            "Prompt": prompt,
            "Matched Keywords": matched,
            "Total Keywords": len(keywords),
            "Coverage (%)": round(coverage * 100, 2),
            "Pass/Fail": status,
            "Raw Response": response_text
        })

    return pd.DataFrame(results)

# ------------------ ✅ RUN & SAVE ALL ------------------ #
semantic_df = evaluate_semantic(semantic_data)
numerical_df = evaluate_numerical(numerical_data)
keyword_df = evaluate_keywords(keyword_data)

# # Combine all
# final_results_df = pd.concat([semantic_df, keyword_df], ignore_index=True)
# final_results_df.to_csv("evaluation_all_results.csv", index=False)
# # Save individual results
# semantic_df.to_csv("evaluated_semantic_results.csv", index=False)
# # numerical_df.to_csv("evaluated_numerical_results.csv", index=False)
# keyword_df.to_csv("evaluated_keyword_results.csv", index=False)

# print("✅ Evaluation complete. Results saved to CSV files.")

# --- Save to a single Excel file with multiple sheets ---
output_path = "agent_results_evaluated.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    numerical_df.to_excel(writer, index=False, sheet_name="Numerical_Eval")
    semantic_df.to_excel(writer, index=False, sheet_name="Semantic_Eval")
    keyword_df.to_excel(writer, index=False, sheet_name="Keyword_Eval")
print(f"\n✅ All evaluations saved to {output_path}")
