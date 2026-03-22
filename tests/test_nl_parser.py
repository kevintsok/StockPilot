"""
Unit tests for the nl_parser module (LLM-based query parsing).
"""

import pytest
from unittest.mock import MagicMock, patch

from auto_select_stock.web.screener import ScreenCriteria
from auto_select_stock.llm.nl_parser import parse_nl_query_with_llm


class MockLLMClient:
    """A mock LLM client that returns a fixed JSON response."""

    def __init__(self, response_json: dict):
        self._response = response_json
        self._model = "test-model"
        self._client = MagicMock()

    @property
    def model(self):
        return self._model

    @property
    def client(self):
        return self._client

    @property
    def score(self):
        """Required by LLMClient ABC but not used by nl_parser."""
        return None

    def _mock_chat_response(self, messages=None, **kwargs):
        """Simulate LLM response."""
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        import json

        mock_choice.message.content = json.dumps(self._response)
        mock_resp.choices = [mock_choice]
        return mock_resp


class TestParseNlQueryWithLlm:
    def test_basic_query_parsed(self):
        """LLM returns correct ScreenCriteria from natural language."""
        mock = MockLLMClient(
            {
                "lookback_days": 90,
                "min_pct_change": None,
                "max_pct_change": 10.0,
                "min_roe": None,
                "min_eps": None,
                "min_turnover_rate": None,
                "max_turnover_rate": None,
            }
        )
        mock.client.chat.completions.create = mock._mock_chat_response

        criteria = parse_nl_query_with_llm("帮我选3个月内涨幅不超过10%的股票", llm_client=mock)

        assert criteria.lookback_days == 90
        assert criteria.max_pct_change == 10.0
        assert criteria.min_pct_change is None

    def test_all_fields_parsed(self):
        """All criteria fields are correctly extracted from LLM response."""
        mock = MockLLMClient(
            {
                "lookback_days": 60,
                "min_pct_change": 5.0,
                "max_pct_change": 20.0,
                "min_roe": 10.0,
                "min_eps": 0.5,
                "min_turnover_rate": 1.0,
                "max_turnover_rate": 5.0,
            }
        )
        mock.client.chat.completions.create = mock._mock_chat_response

        criteria = parse_nl_query_with_llm("涨幅5%~20%，ROE>10%，换手率1%~5%", llm_client=mock)

        assert criteria.lookback_days == 60
        assert criteria.min_pct_change == 5.0
        assert criteria.max_pct_change == 20.0
        assert criteria.min_roe == 10.0
        assert criteria.min_eps == 0.5
        assert criteria.min_turnover_rate == 1.0
        assert criteria.max_turnover_rate == 5.0

    def test_llm_failure_falls_back_to_regex(self):
        """If LLM call fails, regex parser is used as fallback."""

        class FailingClient:
            def __init__(self):
                self._model = "test"
                self._client = MagicMock()

            @property
            def model(self):
                return self._model

            @property
            def client(self):
                return self._client

            def broken_chat(self, messages):
                raise RuntimeError("LLM API error")

        mock = FailingClient()
        mock.client.chat.completions.create = mock.broken_chat

        # Should not raise, should fall back to regex
        criteria = parse_nl_query_with_llm("ROE高于10%", llm_client=mock)
        assert criteria.min_roe == 10.0
        assert criteria.lookback_days == 60

    def test_json_extraction_from_text(self):
        """LLM returns text with surrounding explanation, not just raw JSON."""
        import json

        mock = MockLLMClient(
            {
                "lookback_days": 60,
                "min_pct_change": 10.0,
                "max_pct_change": None,
                "min_roe": None,
                "min_eps": None,
                "min_turnover_rate": None,
                "max_turnover_rate": None,
            }
        )

        def mock_with_wrapper(messages):
            resp = MagicMock()
            choice = MagicMock()
            # LLM returns explanation + JSON
            choice.message.content = (
                "好的，我来帮你解析这个查询。\n"
                '{"lookback_days": 60, "min_pct_change": 10.0, "max_pct_change": null, "min_roe": null, "min_eps": null, "min_turnover_rate": null, "max_turnover_rate": null}\n'
                "以上就是解析结果。"
            )
            resp.choices = [choice]
            return resp

        mock.client.chat.completions.create = mock_with_wrapper

        criteria = parse_nl_query_with_llm("涨幅超过10%", llm_client=mock)
        assert criteria.min_pct_change == 10.0
        assert criteria.lookback_days == 60

    def test_null_fields_ignored(self):
        """Fields with null values are properly handled."""
        mock = MockLLMClient(
            {
                "lookback_days": 60,
                "min_pct_change": None,
                "max_pct_change": None,
                "min_roe": 10.0,
                "min_eps": None,
                "min_turnover_rate": None,
                "max_turnover_rate": None,
            }
        )
        mock.client.chat.completions.create = mock._mock_chat_response

        criteria = parse_nl_query_with_llm("ROE高于10%", llm_client=mock)

        assert criteria.min_roe == 10.0
        assert criteria.min_pct_change is None
        assert criteria.max_pct_change is None
        assert criteria.min_eps is None
