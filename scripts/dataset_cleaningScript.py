import os
import pandas as pd

def convert_to_sample_format(sample_file, target_folder, output_folder):
    print("🔍 Loading sample file...")
    sample_df = pd.read_excel(sample_file)
    sample_columns = sample_df.columns
    print("✅ Sample loaded. Columns:", list(sample_columns))

    os.makedirs(output_folder, exist_ok=True)

    # Walk through all subdirectories
    for root, _, files in os.walk(target_folder):
        for filename in files:
            if filename.endswith(".xlsx") and filename.lower() != os.path.basename(sample_file).lower():
                file_path = os.path.join(root, filename)
                print(f"\n📄 Processing file: {file_path}")

                try:
                    df = pd.read_excel(file_path)
                    print("🔹 First few rows of input:")
                    print(df.head())

                    converted_df = pd.DataFrame()

                    for i, col in enumerate(sample_columns):
                        if col in df.columns:
                            converted_df[col] = df[col]
                        elif i < len(df.columns):
                            print(f"⚠️ Column '{col}' not found, mapping by position.")
                            converted_df[col] = df.iloc[:, i]
                        else:
                            print(f"⚠️ Column '{col}' not found and no fallback — filling with empty.")
                            converted_df[col] = pd.NA

                    # Save using original filename, but into output folder
                    output_name = os.path.splitext(filename)[0] + ".csv"
                    output_path = os.path.join(output_folder, output_name)
                    converted_df.to_csv(output_path, index=False)
                    print(f"✅ Saved cleaned file to: {output_path}")

                except Exception as e:
                    print(f"❌ ERROR processing {filename}: {e}")
            else:
                print(f"⏭️ Skipping {filename} (not xlsx or it's the sample)")

if __name__ == "__main__":
    SAMPLE_FILE = r"C:\Users\aafta\Desktop\CarbonSense-powered-by-IBM-Watsonx\Data_processed\electricity\electricity Canada.xlsx"
    TARGET_FOLDER = r"C:\Users\aafta\Desktop\CarbonSense-powered-by-IBM-Watsonx\Data_processed"
    OUTPUT_FOLDER = r"C:\Users\aafta\Desktop\CarbonSense-powered-by-IBM-Watsonx\Converted"

    convert_to_sample_format(SAMPLE_FILE, TARGET_FOLDER, OUTPUT_FOLDER)
