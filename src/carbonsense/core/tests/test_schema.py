from json import loads
from jsonschema import validate
from ...config.common_schema import carbon_metric as schema

def test_sample_metric():
    sample = '{"value":2.5,"emission_unit":"kg CO2e","product_unit":"per kg","source":"milvus","confidence":0.84,"product_name":"beef","category":"Food and Beverages"}'
    validate(loads(sample), schema) 