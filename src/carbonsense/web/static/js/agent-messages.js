// Shared agent message constants and functions
const AGENT_MESSAGES = {
    "query_classifier": "Classifying your query to understand your intent and category...",
    "entity_extractor": "Extracting key entities and details from your question...",
    "unit_normalizer": "Normalizing all quantities and units for consistency...",
    "footprint_cache": "Checking the cache for existing carbon footprint data...",
    "milvus_researcher": "Searching the Milvus database for relevant carbon data...",
    "discovery_researcher": "Searching Watson Discovery for scientific carbon data...",
    "serper_researcher": "Searching the web for recent carbon footprint information...",
    "unit_harmoniser": "Harmonizing all carbon metrics into a standard format...",
    "carbon_estimator": "Calculating precise carbon footprints using the best data...",
    "metric_ranker": "Evaluating and ranking all carbon metrics for reliability...",
    "comparison_formatter": "Comparing carbon footprints to highlight differences...",
    "recommendation_agent": "Generating personalized recommendations to reduce your footprint...",
    "explanation_agent": "Providing clear explanations and debunking myths...",
    "answer_formatter": "Formatting the final answer for you...",
    "answer_consolidator": "Consolidating all information into a cohesive response...",
    "manager": "Managing the analysis workflow...",
    "intent_classifier": "Determining the intention behind your query...",
    "entity_recognition": "Identifying key entities in your question...",
    "activity_processor": "Processing your activity details...",
    "footprint_calculator": "Calculating accurate carbon footprints...",
    "recommendations_generator": "Generating tailored recommendations...",
    "data_collector": "Collecting relevant carbon data sources...",
    "done": "Analysis complete! Preparing final response..."
};

function getAgentMessage(agent) {
    return AGENT_MESSAGES[agent] || `Processing step: ${agent}`;
}

// Export to global scope for use in other modules
window.AGENT_MESSAGES = AGENT_MESSAGES;
window.getAgentMessage = getAgentMessage; 