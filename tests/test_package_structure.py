import importlib.util
import unittest


NEW_IMPORT_PATHS = [
    "agisample.common.elasticsearch_connection",
    "agisample.langchain.agents.sample_agent_process",
    "agisample.langchain.agents.sample_agent_process_by_json",
    "agisample.langchain.rag.sample_rag_process",
    "agisample.langchain.vectorstores.sample_data_vector_manager",
    "agisample.langchain.vectorstores.sample_data_es_manager",
    "agisample.langchain.sql.sample_sql_process",
    "agisample.langchain.multimodal.sample_image_process",
    "agisample.langchain.extraction.sample_structured_output_process",
    "agisample.agentscope.sample_agentscope",
    "agisample.document_ai.azure_document_intelligence.prebuilt_read",
    "agisample.document_ai.azure_document_intelligence.prebuilt_invoice",
    "agisample.document_ai.resume_downloader.download_resume_file",
    "agisample.local_models.mini_cpm_rag",
    "agisample.integrations.langflow_process",
    "agisample.machine_learning.dimensionality_reduction.sample_reduce_dimension_process",
    "agisample.langgraph.a2a.llm_factory",
    "agisample.langgraph.a2a.peer_tool",
    "agisample.langgraph.a2a.agent_executor",
    "agisample.langgraph.a2a.agents",
    "agisample.langgraph.a2a.server_a",
    "agisample.langgraph.a2a.server_b",
    "agisample.langgraph.a2a.client",
    "agisample.langgraph.basic.sample_graph_state",
    "agisample.langgraph.basic.sample_graph_basic_chatbot",
    "agisample.langgraph.basic.sample_graph_chatbot",
    "agisample.langgraph.basic.sample_graph_process",
    "agisample.langgraph.basic.sample_graph_add_human_feedback",
    "agisample.langgraph.customersupport.llm_model",
    "agisample.langgraph.customersupport.sample_graph_customer_support_bot_final",
    "agisample.langgraph.hierarchical_teams.main_team",
    "agisample.langgraph.hierarchical_teams.web_tool",
    "agisample.langchain.extraction.llm_multi_extract_process",
    "agisample.langchain.extraction.lang_extract_process",
    "agisample.langchain.code.llm_generate_code_process",
    "agisample.langchain.code.llm_review_code_process",
    "agisample.langchain.document.llm_file_process",
    "agisample.langchain.document.llm_prompt_process",
    "agisample.machine_learning.encode_only_process",
    "agisample.machine_learning.random_forest_process",
    "agisample.tools.find_combinations",
    "agisample.tools.recovery_to_markdown",
    "agisample.tools.email_info_process",
]


LEGACY_IMPORT_PATHS = [
    "agisample.base.ElasticsearchConnection",
    "agisample.generic.SampleAzureDocIntelligencePrebuiltRead",
    "agisample.generic.SampleAzureDocIntelligencePrebuiltInvoice",
    "agisample.generic.SampleDownloadResumeFile",
    "agisample.generic.mini_cpm_rag",
    "agisample.generic.SampleLangflowProcess",
    "agisample.generic.SampleReduceDimensionProcess",
]


class PackageStructureTest(unittest.TestCase):
    def test_new_demo_package_paths_exist(self):
        for module_name in NEW_IMPORT_PATHS:
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_legacy_demo_package_paths_remain_available(self):
        for module_name in LEGACY_IMPORT_PATHS:
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))


if __name__ == "__main__":
    unittest.main()
