# Data Processing Guide for CarbonSense

## 📑 Column Descriptions

| Column Name    | Description                                                                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `industry`   | Specifies the sector or category of electricity production. This may include grid mix, coal-based generation, renewables, etc.                                   |
| `unit`       | The measurement unit used for impact values, typically**MJ** (megajoules) or **kg CO₂e** (kilograms of CO₂ equivalent).                            |
| `process`    | Describes the electricity production process, usually including the U.S. state name (e.g.,`Electricity Texas production`).                                     |
| `total`      | The total value for the entire process. May represent energy generated or cumulative environmental impact.                                                       |
| `carbon`     | Total greenhouse gas emissions related to the process, expressed in**kg CO₂-equivalent**. Includes CO₂, CH₄, N₂O using global warming potential (GWP). |
| `ced`        | **Cumulative Energy Demand** — the total amount of direct and indirect energy (in MJ) required throughout the process life cycle.                         |
| `region`     | The broader geographic region. In this dataset, this is consistently `north_america`.                                                                          |
| `source`     | The source of the data, which may include research papers, government databases, or industry reports.                                                            |
| `confidence` | A score between 0 and 1 indicating the reliability of the data point.                                                                                            |

## 🔍 Use Cases

- ⚙️ Life Cycle Assessment (LCA)
- 🌎 Carbon footprint analysis
- 🔋 Energy efficiency benchmarking
- 🗺️ Regional/state-level energy comparison
- 📊 Product impact comparison
- 🥘 Food carbon footprint calculation
- 🚗 Transportation emissions analysis
- 🏠 Household energy usage assessment

## 💾 Data Processing Flow

1. **Raw Data Ingestion**

   - Original Excel files loaded from various sources
   - Data validated for required columns and formats
   - File metadata extracted for provenance tracking
2. **Normalization**

   - Units standardized across all datasets
   - Column names harmonized
   - Missing values handled appropriately
   - Outliers identified and addressed
3. **Enhancement**

   - Additional context added where available
   - Confidence scores calculated based on source reliability
   - Regional information expanded
   - Cross-referencing between different data sources
4. **Storage**

   - Processed files organized by region and industry
   - Vector embeddings generated for semantic search
   - Metadata indexed for efficient retrieval
   - Versioning implemented for data lineage
