import pytest
from unittest.mock import Mock, patch
from ....carbonsense.core.rag_generator import RAGGenerator

@pytest.fixture
def mock_config():
    return Mock()

@pytest.fixture
def mock_milvus():
    with patch("carbonsense.services.milvus_service.MilvusService") as mock:
        yield mock

@pytest.fixture
def mock_watsonx():
    with patch("carbonsense.services.watsonx_service.WatsonxService") as mock:
        yield mock

@pytest.fixture
def rag_generator(mock_config, mock_milvus, mock_watsonx):
    return RAGGenerator(mock_config)

def test_generate_answer_success(rag_generator, mock_milvus, mock_watsonx):
    # Setup
    query = "What is carbon footprint?"
    mock_embedding = [0.1, 0.2, 0.3]
    mock_search_results = [
        Mock(entity={"chunk_text": "Carbon footprint is...", "file_name": "doc1.txt"})
    ]
    mock_response = "Carbon footprint is the total amount of greenhouse gases..."

    mock_watsonx.generate_embeddings.return_value = [mock_embedding]
    mock_milvus.search_vectors.return_value = mock_search_results
    mock_watsonx.generate_text.return_value = mock_response

    # Execute
    result = rag_generator.generate_answer(query)

    # Assert
    assert result == mock_response
    mock_watsonx.generate_embeddings.assert_called_once_with([query])
    mock_milvus.search_vectors.assert_called_once()
    mock_watsonx.generate_text.assert_called_once()

def test_generate_answer_no_context(rag_generator, mock_milvus, mock_watsonx):
    # Setup
    query = "What is carbon footprint?"
    mock_embedding = [0.1, 0.2, 0.3]
    mock_search_results = []

    mock_watsonx.generate_embeddings.return_value = [mock_embedding]
    mock_milvus.search_vectors.return_value = mock_search_results

    # Execute
    result = rag_generator.generate_answer(query)

    # Assert
    assert result == "I cannot find information about that in the available data."
    mock_watsonx.generate_text.assert_not_called()

def test_get_context_success(rag_generator, mock_milvus, mock_watsonx):
    # Setup
    query = "What is carbon footprint?"
    mock_embedding = [0.1, 0.2, 0.3]
    mock_search_results = [
        Mock(
            entity={"chunk_text": "Carbon footprint is...", "file_name": "doc1.txt"},
            score=0.95
        )
    ]

    mock_watsonx.generate_embeddings.return_value = [mock_embedding]
    mock_milvus.search_vectors.return_value = mock_search_results

    # Execute
    result = rag_generator.get_context(query)

    # Assert
    assert len(result) == 1
    assert result[0]["text"] == "Carbon footprint is..."
    assert result[0]["file_name"] == "doc1.txt"
    assert result[0]["score"] == 0.95 