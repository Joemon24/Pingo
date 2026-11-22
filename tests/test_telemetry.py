from unittest.mock import Mock

from app.telemetry import execution, model_run


def test_trace_records_usage_without_customer_text():
    token = execution.set([])
    try:
        agent = Mock()
        agent.run_sync.return_value = Mock(
            output="draft", usage=Mock(input_tokens=12, output_tokens=8),
        )
        assert model_run(agent, "private customer text", "generation") == "draft"
        stages = execution.get()
        assert stages[0]["input_tokens"] == 12
        assert stages[0]["output_tokens"] == 8
        assert "private" not in str(stages)
        assert stages[0]["latency_ms"] >= 0
    finally:
        execution.reset(token)
