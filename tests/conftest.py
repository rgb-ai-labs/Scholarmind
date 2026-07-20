import pytest

# Modules whose every test loads a HuggingFace embedding/reranker model (via ingestion or search).
_SLOW_MODULES = {
    "test_agents_base",
    "test_dense",
    "test_delete",
    "test_discovery",
    "test_gap_analysis",
    "test_ingestion_dedupe",
    "test_methodology",
    "test_novelty",
    "test_papers",
    "test_pipeline",
    "test_qa",
    "test_reranker",
    "test_search",
    "test_sparse",
    "test_summarization",
    "test_writing",
}

# Modules whose every test makes a live network (Crossref) call.
_LLM_MODULES = {
    "test_citation_service",
}

# Individual tests in otherwise-fast modules that load a model.
_SLOW_TESTS = {
    "test_ingest_with_path",
    "test_ask_returns_expected_schema_with_mocked_llm",
    "test_ask_no_sources_returns_empty_answer",
    "test_run_eval_end_to_end",
    "test_graph_routes_to_agent_node",
    "test_graph_final_node_formats_and_verifies_citations",
    "test_graph_formatting_failure_preserves_answer",
    "test_graph_ingest_path",
    "test_graph_ask_path",
    "test_graph_messages_accumulate",
    "test_run_ingest_path",
    "test_run_ask_path",
    "test_cli_chat_ingest_command",
    "test_ingest_candidate_metadata_only_record_is_searchable_end_to_end",
}

# Individual tests that make a live LLM or network call (each is also skipif-guarded).
_LLM_TESTS = {
    "test_openrouter_client_real_call_returns_pong",
    "test_verify_claim_support_real_llm_flags_mismatched_claim",
    "test_normalize_metadata_real_paper_matches_crossref",
    "test_answer_question_real_end_to_end",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        name = item.name.split("[", 1)[0]
        if module in _SLOW_MODULES or name in _SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if module in _LLM_MODULES or name in _LLM_TESTS:
            item.add_marker(pytest.mark.llm)
