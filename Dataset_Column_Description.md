# 📊 Dataset Schema: Cleaned Environmental & Energy Data

This schema applies to all Excel files in the Data_processed folder. Each file contains standardized environmental impact and energy consumption data processed from the original datasets.

## Column Descriptions

| Column Name                | Description                                                                 | Data Type              |
|----------------------------|-----------------------------------------------------------------------------|------------------------|
| `industry`                 | Industry or sector (e.g., electricity Canada, food, chemicals)              | Categorical (text)     |
| `unit`                     | Measurement unit for the values (e.g., MJ, kg, kWh)                         | Various (e.g. kg, kWh) |
| `process`                  | Specific process (e.g., Electricity Alberta production)                     | Categorical (text)     |
| `carbon (kg CO2 eq)`       | Carbon footprint measured in kilograms of CO₂ equivalent                    | Carbon footprint       |
| `ced (MJ)`                 | Cumulative Energy Demand measured in Megajoules                             | Cumulative energy demand |
| `global warming (kg CO2 eq)` | Global warming potential measured in kg CO₂ equivalent                     | Global warming potential |
| `climate change (kg CO2 eq)` | Climate change impact measured in kg CO₂ equivalent                        | Climate change impact  |
| `region`                   | Geographic region associated with the data (e.g., Global, EU, USA)          | Categorical (text)     |

## Dataset Organization

The processed data is organized into four main directories:

### 1. Regional Data (`/regional`)
Contains region-specific consolidated datasets:
- `asia_consolidated.xlsx` - Data for Asian countries
- `europe_consolidated.xlsx` - Data for European countries
- `north_america_consolidated.xlsx` - Data for North American countries
- `rest_of_world_consolidated.xlsx` - Data for all other regions

### 2. Industry-Specific Data (`/industry`)
Contains 25 industry-specific datasets including:
- Manufacturing sectors (metals, electronics, textiles, etc.)
- Food and agriculture
- Building materials
- Chemicals
- Energy and fuels
- End-of-life processing
- Transportation

### 3. Electricity Data (`/electricity`)
Contains electricity production and consumption data by region:
- `electricity Canada.xlsx` - Electricity production data across Canadian provinces
- `electricity China.xlsx` - Electricity data for China
- `electricity EU.xlsx` - Electricity data for European Union countries
- `electricity India.xlsx` - Electricity data for India
- `electricity USA.xlsx` - Electricity data for the United States
- `electricity Rest of the World.xlsx` - Electricity data for other regions
- `electricity general industry.xlsx` - General electricity consumption data

### 4. Metadata (`/metadata`)
Contains supporting information:
- `.heading.xlsx` - Header and attribute definitions

## Data Processing

The dataset undergoes several processing steps performed by the `preprocess_dataset.py` script:

1. **Data Cleaning**
   - Standardization of column names and units (e.g., carbon → carbon (kg CO2 eq))
   - Filtering to keep only the most relevant columns
   - Adding data type annotations as Excel comments
   - Normalization of text fields

2. **Semantic Enhancement**
   - Generation of vector embeddings for each data chunk
   - Storage in Milvus vector database
   - Metadata tagging for improved retrieval

3. **Usage in System**
   - Retrieved during user queries about carbon footprints
   - Combined with web search results when needed
   - Validated against multiple sources for accuracy
   - Cached to improve performance of repeated queries

## Data Sources

The datasets are compiled from multiple reliable sources including:

- Environmental Protection Agency (EPA)
- Department of Energy (DOE)
- Academic research publications
- Industry sustainability reports
- International environmental organizations

For questions about this dataset or to contribute additional data, please contact the project team.
