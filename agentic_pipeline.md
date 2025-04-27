# CarbonSense Agentic Pipeline

## Pipeline Overview

The CarbonSense system uses a modular, multi-agent pipeline to process sustainability-related queries. This document provides a visual representation of the agent flow.

```
                                        +------------------------+
                                        |                        |
                                        |   Query from User      |
                                        |                        |
                                        +------------+-----------+
                                                    |
                                                    v
                           +-----------------------------------------------------------+
                           |                                                           |
                           |                1. QUERY UNDERSTANDING LAYER               |
                           |                                                           |
+----------+               |  +----------------+     +----------------+     +--------+ |
|          |               |  |                |     |                |     |        | |
|  Cache   |<--------------+--| Query          |---->| Entity         |---->| Unit   | |
|  Lookup  |               |  | Classifier     |     | Extractor      |     | Norm.  | |
|          |               |  |                |     |                |     |        | |
+-----+----+               |  +----------------+     +----------------+     +--------+ |
      |                    |                                                           |
      |                    +-----------------------------------------------------------+
      |                                        |
      |                                        v
      |                    +-----------------------------------------------------------+
      |                    |                                                           |
      |                    |              2. RETRIEVAL & RESEARCH LAYER                |
      |                    |                                                           |
      |                    | +------------+    +-------------+    +-------------+      |
      |                    | |            |    |             |    |             |      |
      |  If Cache MISS ------>  Milvus    |--->| Watson      |--->| Serper      |---+  |
      |                    | | Researcher |    | Discovery   |    | Web Search  |   |  |
      |                    | |            |    | Researcher  |    | Researcher  |   |  |
      |                    | +------------+    +-------------+    +-------------+   |  |
      |                    |                                                        |  |
      |                    +-----------------------------------------------------------+
      |                                                                             |
      |                                                                             v
      |                    +-----------------------------------------------------------+
      |                    |                                                           |
      |                    |           3. CARBON ESTIMATION & SYNTHESIS                |
      |                    |                                                           |
      |                    | +------------+    +-------------+    +-------------+      |
      |                    | |            |    |             |    |             |      |
      +--------------------->| Harmoniser |--->| Carbon      |--->| Metric      |---+  |
      If Cache HIT         | |            |    | Estimator   |    | Ranker      |   |  |
                           | +------------+    +-------------+    +-------------+   |  |
                           |                                                        |  |
                           +-----------------------------------------------------------+
                                                                                    |
                                                                                    v
                          +-----------------------------------------------------------+
                          |                                                           |
                          |            4. INTENT-SPECIFIC PROCESSING                  |
                          |                                                           |
"estimate" --------------->                                                           |
                          |                                                           |
"compare" ---------------->  +------------+    +-------------+    +-------------+     |
                          |  |            |    |             |    |             |     |
"suggest" ---------------->  | Comparison |    | Recommend.  |    | Explanation |     |
                          |  | Formatter  |    | Agent       |    | Agent       |     |
"lifecycle" -------------->  |            |    |             |    |             |     |
                          |  +------------+    +-------------+    +-------------+     |
"myth_bust" -------------->                                                           |
                          |                                                           |
                          +-----------------------------+-----------------------------+
                                                       |
                                                       v
                          +-----------------------------------------------------------+
                          |                                                           |
                          |                5. RESPONSE GENERATION                     |
                          |                                                           |
                          |                   +----------------+                      |
                          |                   |                |                      |
                          |                   | Answer         |                      | 
                          |                   | Formatter      |                      |
                          |                   |                |                      |
                          |                   +----------------+                      |
                          |                                                           |
                          +-----------------------------------------------------------+
                                           |
                                           v
                          +-----------------------------------------------------------+
                          |                                                           |
                          |                 Response to User                          |
                          |                                                           |
                          |   {                                                       |
                          |     "emission": "12.5 kg CO₂e",                           |
                          |     "method": "Includes emissions from manufacturing...", |
                          |     "category": "Purchases",                              |
                          |     "sources": [                                          |
                          |       {"title": "Source 1", "url": "https://..."},        |
                          |       {"title": "Source 2", "url": "https://..."}         |
                          |     ]                                                     |
                          |   }                                                       |
                          |                                                           |
                          +-----------------------------------------------------------+
```

## Intent-Based Agent Flow

The system supports multiple query intents:

1. **Carbon Emission Estimation** (Intent: `estimate`)
   - QueryClassifier → EntityExtractor → Research Agents → Estimator → Formatter

2. **Comparison** (Intent: `compare`)
   - QueryClassifier → Dual EntityExtractor → Parallel Retrieval & Estimation → Comparison Formatter

3. **Suggestion** (Intent: `suggest`)
   - QueryClassifier → Retrieval Agents → Estimator → RecommendationAgent → Formatter

4. **Lifecycle Analysis** (Intent: `lifecycle`)
   - QueryClassifier → EntityExtractor → Retrieval Agents → Estimator (lifecycle mode) → Formatter

5. **Myth Busting / Fact Check** (Intent: `myth_bust`)
   - QueryClassifier → Retriever Agents → Estimator → ExplanationAgent → Formatter

## Category Classification

All outputs are classified into one of five key categories:

1. **Food & Diet**
   - Keywords: food, meal, ingredient, tofu, milk, latte, red meat, banana, coffee

2. **Energy Use**
   - Keywords: AC, appliance, TV, electricity, kWh, shower, heater, lighting, standby

3. **Mobility**
   - Keywords: drive, bus, bike, Uber, EV, train, flight, carpool, commute

4. **Purchases**
   - Keywords: iPhone, tote bag, Amazon, fashion, clothes, streaming, recycling, device

5. **Miscellaneous**
   - Abstract or mixed activities, combined behaviors, complex lifestyle changes

## Output Format

All responses follow a standardized JSON format:

```json
{
  "emission": "12.5 kg CO₂e",
  "method": "Includes emissions from manufacturing, delivery, and disposal. Based on NREL lifecycle database.",
  "category": "Purchases",
  "sources": [
    {"title": "NREL Product LCA", "url": "https://nrel.gov/..."},
    {"title": "GHG Protocol", "url": "https://ghgprotocol.org/..."}
  ]
}
```

For specialized intents like `compare`, `suggest`, or `myth_bust`, additional fields are included in the response. 