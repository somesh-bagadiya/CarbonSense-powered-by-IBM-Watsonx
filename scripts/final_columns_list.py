# -----------------------------------------------
# generate_shortlist_columns.py
# -----------------------------------------------
def main():
    # Define the key columns and their default units.
    # If a column has no inherent measurement, we write "N/A" or "Categorical" to clarify.
    # If units could vary by context, we use "Various" to acknowledge that the data may vary.
    
    # Common column definitions reused for nearly all files:
    common_columns = [
        {"col_num": None, "header": "industry",       "units": "Categorical (text)"},
        {"col_num": None, "header": "unit",           "units": "Various (e.g. kg, kWh)"},
        {"col_num": None, "header": "process",        "units": "Categorical (text)"},
        {"col_num": None, "header": "carbon",         "units": "kg CO2 eq"},
        {"col_num": None, "header": "ced",            "units": "MJ (cumulative energy demand)"},
        {"col_num": None, "header": "global warming,", "units": "kg CO2 eq"},
        {"col_num": None, "header": "climate change", "units": "kg CO2 eq"},
        {"col_num": None, "header": "region",         "units": "Categorical (text)"}
    ]

    # Now define each file, specifying whichever columns or notes differ:
    files_info = [
        # 1. Electricity files
        {
            "filename": "electricity Canada.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity China.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity EU.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity general industry.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity India.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity Rest of the World.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electricity USA.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        # 2. Industry files
        {
            "filename": "agriculture.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "building materials.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "ceramics.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "chem proxi.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "chemicals.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "electronics.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "end-of-life.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "fibres.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "food.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "Food_Global.xlsx",
            "sheet": "Sheet1",
            "columns": [
                # Example: this file has "Unit" and "Process" capitalized
                {"col_num": None, "header": "industry",        "units": "Categorical (text)"},
                {"col_num": None, "header": "Unit",            "units": "Various (e.g. kg, kWh)"},
                {"col_num": None, "header": "Process",         "units": "Categorical (text)"},
                {"col_num": None, "header": "Carbon",          "units": "kg CO2 eq"},
                {"col_num": None, "header": "CED",             "units": "MJ (cumulative energy demand)"},
                {"col_num": None, "header": "Global warming,", "units": "kg CO2 eq"},
                {"col_num": None, "header": "Climate change",  "units": "kg CO2 eq"},
                {"col_num": None, "header": "region",          "units": "Categorical (text)"}
            ]
        },
        {
            "filename": "Food_USA.xlsx",
            "sheet": "Sheet1",
            "columns": [
                {"col_num": None, "header": "industry",        "units": "Categorical (text)"},
                {"col_num": None, "header": "Unit",            "units": "Various (e.g. kg, kWh)"},
                {"col_num": None, "header": "Process",         "units": "Categorical (text)"},
                {"col_num": None, "header": "Carbon",          "units": "kg CO2 eq"},
                {"col_num": None, "header": "CED",             "units": "MJ (cumulative energy demand)"},
                {"col_num": None, "header": "Global warming,", "units": "kg CO2 eq"},
                {"col_num": None, "header": "Climate change",  "units": "kg CO2 eq"},
                {"col_num": None, "header": "region",          "units": "Categorical (text)"}
            ]
        },
        {
            "filename": "fuels.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "glass.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "heat.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "laminates.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "metals, ferro.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "metals, non-ferro.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "paper & packaging.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        },
        {
            "filename": "plastics.xlsx",
            "sheet": "Sheet1",
            "columns": common_columns
        }
    ]

    # Write the information into a text file
    with open("shortlist_of_columns.txt", "w", encoding="utf-8") as f:
        for file_info in files_info:
            f.write(f"File: {file_info['filename']}\n")
            f.write(f"Sheet: {file_info['sheet']}\n")
            f.write("Columns:\n")
            for idx, col in enumerate(file_info["columns"], start=1):
                # If col_num is None, we can simply use idx to list them in order
                col_name = col["header"]
                col_units = col["units"]
                f.write(f"  {idx}. {col_name} ({col_units})\n")
            f.write("\n" + "="*80 + "\n\n")


if __name__ == "__main__":
    main()
