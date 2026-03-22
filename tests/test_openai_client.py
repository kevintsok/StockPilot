"""
Unit tests for the OpenAI client module (llm/openai_client.py).

Tests LLM client with mocked HTTP responses.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auto_select_stock.core.types import StockDailyRow, StockMeta, StockSnapshot, StockScore
from auto_select_stock.llm.openai_client import OpenAIClient


class TestOpenAIClientParseScore:
    """Tests for _parse_score()."""

    def test_parses_integer_score(self):
        """Parses integer score from text."""
        result = OpenAIClient._parse_score("The score is 8")
        assert result == 8.0

    def test_parses_float_score(self):
        """Parses float score from text."""
        result = OpenAIClient._parse_score("Score: 7.5")
        assert result == 7.5

    def test_parses_score_with_surrounding_text(self):
        """Parses score from text with surrounding content."""
        result = OpenAIClient._parse_score("Based on analysis, the score is 8.5 out of 10.")
        assert result == 8.5

    def test_clamps_high_scores(self):
        """Scores above 10 are clamped to 10."""
        result = OpenAIClient._parse_score("Score: 15")
        assert result == 10.0

    def test_clamps_negative_scores(self):
        """Scores below 0 are clamped to 0."""
        result = OpenAIClient._parse_score("Score: -3")
        assert result == 0.0

    def test_returns_zero_on_no_match(self):
        """Returns 0.0 when no numeric score found."""
        result = OpenAIClient._parse_score("No score in this text")
        assert result == 0.0

    def test_parses_first_match(self):
        """Returns first match when multiple numbers present."""
        result = OpenAIClient._parse_score("7.5 is the score, but also 9.0 mentioned")
        assert result == 7.5


class TestOpenAIClientInit:
    """Tests for OpenAIClient initialization."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    def test_init_with_openai_provider(self):
        """Initializes with OpenAI provider using env var."""
        client = OpenAIClient(provider="openai")
        assert client.provider == "openai"
        assert client.model == "gpt-4o-mini"  # default

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_init_with_custom_model(self):
        """Allows custom model override."""
        client = OpenAIClient(model="gpt-4o", provider="openai")
        assert client.model == "gpt-4o"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_init_with_custom_base_url(self):
        """Allows custom base URL override."""
        client = OpenAIClient(base_url="https://custom.api.com/v1", provider="openai")
        assert client._base_url == "https://custom.api.com/v1"

    def test_raises_without_api_key(self):
        """Raises RuntimeError when no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
                OpenAIClient(provider="openai")


class TestOpenAIClientScore:
    """Tests for score() method."""

    def _make_snapshot(self) -> StockSnapshot:
        from datetime import datetime
        row = StockDailyRow(
            date=datetime(2024, 1, 1),
            open=10.0, high=10.5, low=9.5, close=10.2,
            volume=1000000, amount=10200000, turnover_rate=2.5,
            volume_ratio=1.2, pct_change=2.1, amplitude=5.0, change_amount=0.2,
        )
        return StockSnapshot(
            meta=StockMeta(symbol="600000", name="TestStock"),
            recent_rows=[row],
            factors={"roe": 15.0, "pe": 8.0},
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch.object(OpenAIClient, "__init__", lambda self, **kwargs: None)
    def test_score_returns_stock_score(self):
        """Returns StockScore with symbol and parsed score."""
        client = OpenAIClient()
        client._provider = "openai"
        client._model = "gpt-4o-mini"
        client._api_key = "test-key"
        client._base_url = "https://api.openai.com/v1"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "8.5"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client._client = mock_client

        snapshot = self._make_snapshot()
        result = client.score(snapshot)

        assert isinstance(result, StockScore)
        assert result.symbol == "600000"
        assert result.score == 8.5

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch.object(OpenAIClient, "__init__", lambda self, **kwargs: None)
    def test_score_parses_numeric_response(self):
        """Correctly parses numeric response from LLM."""
        client = OpenAIClient()
        client._provider = "openai"
        client._model = "gpt-4o-mini"
        client._api_key = "test-key"
        client._base_url = "https://api.openai.com/v1"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The score is 7"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client._client = mock_client

        result = client.score(self._make_snapshot())
        assert result.score == 7.0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch.object(OpenAIClient, "__init__", lambda self, **kwargs: None)
    def test_score_handles_empty_content(self):
        """Handles empty response content gracefully."""
        client = OpenAIClient()
        client._provider = "openai"
        client._model = "gpt-4o-mini"
        client._api_key = "test-key"
        client._base_url = "https://api.openai.com/v1"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        client._client = mock_client

        result = client.score(self._make_snapshot())
        assert result.score == 0.0  # defaults to 0 on empty
