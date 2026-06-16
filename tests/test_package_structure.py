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
]


LEGACY_IMPORT_PATHS = [
    "agisample.base.ElasticsearchConnection",
    "agisample.framework.SampleAgentProcess",
    "agisample.framework.SampleAgentProcessByJson",
    "agisample.framework.SampleRagProcess",
    "agisample.framework.SampleDataVectorManager",
    "agisample.framework.SampleDataEsManager",
    "agisample.framework.SampleSQLProcess",
    "agisample.framework.SampleImageProcess",
    "agisample.framework.SampleStructuredOutputProcess",
    "agisample.framework.SampleAgentScope",
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
