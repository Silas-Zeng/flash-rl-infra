import sys

import pytest

from flashrl.rollout import SGLangRolloutBackend
from flashrl.schemas import Prompt


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _prompt() -> Prompt:
    return Prompt("run", "prompt", "group", "", [1, 2], 17, 3)


def test_sglang_generate_contract_and_meta_parsing(monkeypatch):
    calls = []

    class FakeHttpx:
        @staticmethod
        def post(url, *, json, timeout):
            calls.append((url, json, timeout))
            return _Response(
                {
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 11], [-0.2, 12]],
                        "completion_tokens": 2,
                        "finish_reason": {"type": "length", "length": 2},
                        "weight_version": "7",
                    }
                }
            )

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)
    result = SGLangRolloutBackend("http://127.0.0.1:30000", "toy").generate(_prompt(), "trajectory")
    assert calls[0][0] == "http://127.0.0.1:30000/generate"
    assert calls[0][1]["input_ids"] == [1, 2]
    assert calls[0][1]["sampling_params"]["max_new_tokens"] == 128
    assert calls[0][1]["return_logprob"] is True
    assert result.trajectory.response_ids == [11, 12]
    assert result.trajectory.finish_reason == "length"
    assert result.trajectory.policy_version == 7


def test_sglang_adapter_rejects_mismatched_logprob_count(monkeypatch):
    class FakeHttpx:
        @staticmethod
        def post(url, *, json, timeout):
            return _Response(
                {
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 11]],
                        "completion_tokens": 2,
                    }
                }
            )

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)
    with pytest.raises(RuntimeError, match="token count"):
        SGLangRolloutBackend("http://127.0.0.1:30000", "toy").generate(_prompt(), "trajectory")
