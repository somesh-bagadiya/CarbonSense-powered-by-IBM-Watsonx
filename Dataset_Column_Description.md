# 📊 Dataset Schema: Cleaned Environmental & Energy Data

This schema applies to all converted files in the dataset folder. Below is a description of each column.

| Column Name                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `industry`                     | Industry or sector related to the data entry                               |
| `unit`                         | Measurement unit for the values (e.g., MJ, kg CO₂-eq)                      |
| `process`                      | Description of the process involved (e.g., production, consumption)        |
| `metals`                       | Metal-related impact or resource usage                                     |
| `carbon`                       | Carbon-related emission data                                               |
| `ced`                          | Cumulative Energy Demand (total energy input)                              |
| `non-renewable`                | Portion of energy or resources from non-renewable sources                  |
| `renewable`                    | Portion of energy or resources from renewable sources                      |
| `global warming`               | Greenhouse gas emissions contributing to global warming                    |
| `global warming.1`             | Possibly regional or lifecycle phase specific GHG emissions                |
| `ozone formation`              | Potential to contribute to ozone layer formation                          |
| `climate change`               | Broader climate change impact beyond GHGs                                 |
| `photochemical ozone formation`| Impact on smog and low-level ozone formation                              |
| `resource use`                 | General use of material or natural resources                               |
| `carbon.1`                     | Additional or corrected carbon emission data                               |
| `source`                       | Origin of the data (research paper, government database, industry report)  |
| `confidence`                   | Reliability score between 0.0 and 1.0                                      |
| `region`                       | Geographic region associated with the data                                 |
| `timestamp`                    | When the data was recorded or last updated                                 |
| `data_version`                 | Version of the dataset                                                     |

## Data Processing

The dataset undergoes several processing steps before being used in the system:

1. **Data Cleaning**
   - Standardization of unit formats
   - Handling of missing values
   - Removal of duplicate entries
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
