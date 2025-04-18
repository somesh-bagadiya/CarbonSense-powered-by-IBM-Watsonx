import json
import subprocess
import time
from pathlib import Path

# Step 1: Set paths
EVAL_FOLDER = Path(__file__).parent
CARBONSENSE_MAIN = EVAL_FOLDER.parent / "carbonsense" / "main.py"
EVAL_INPUT_FILE = EVAL_FOLDER / "evaluation_data.json"
EVAL_OUTPUT_FILE = EVAL_FOLDER / "agent_evaluation_results.json"
FINAL_ANSWER_TXT = EVAL_FOLDER.parent.parent / "0_final_answer.txt"  # outside /src

# Step 2: Load the evaluation data
with open(EVAL_INPUT_FILE, "r") as f:
    eval_data = json.load(f)

# Step 3: Flatten all prompt entries across categories
all_prompts = []
for category in ["numerical_data", "keywords_data", "semantic_explanation_data"]:
    for entry in eval_data.get(category, []):
        all_prompts.append({
            "prompt": entry["prompt"],
            "category": category,
            "expected": entry.get("expected_co2_kg") or entry.get("expected_keywords") or entry.get("reference_answer")
        })

# Step 4: Loop over prompts and call main.py for each
results = []

print(f"\n🔁 Running {len(all_prompts)} total prompts...\n")
for i, item in enumerate(all_prompts):
    prompt = item["prompt"]
    print(f"🚀 [{i+1}/{len(all_prompts)}] Prompt: {prompt}")

    # Start timer
    start = time.time()

    # Run main.py with subprocess
    command = [
        "python", "-m", "src.carbonsense.main",
        "--mode", "crew_agent",
        "--query", prompt
    ]

    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script crashed for: {prompt}")
        actual = {"error": "Script crashed"}
    else:
        # Read response from 0_final_answer.txt
        if FINAL_ANSWER_TXT.exists():
            try:
                with open(FINAL_ANSWER_TXT, "r") as f:
                    actual_text = f.read().strip()
                    actual = {"response_text": actual_text}
            except Exception as err:
                actual = {"error": f"Read error: {err}"}
        else:
            actual = {"error": "0_final_answer.txt not found"}

    end = time.time()
    print(f"⏱️ Completed in {round(end - start, 2)} seconds\n")

    # Append result
    results.append({
        "prompt": prompt,
        "category": item["category"],
        "expected": item["expected"],
        "actual_response": actual
    })

    # Step 5: Save results to output file
    with open(EVAL_OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

print(f"✅ All done! Results saved to: {EVAL_OUTPUT_FILE}\n")
