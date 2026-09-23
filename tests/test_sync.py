import sys

from flashrl.sync import update_sglang_from_disk


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


def test_sync_publishes_numeric_weight_version(monkeypatch, tmp_path):
    calls = []

    class FakeHttpx:
        @staticmethod
        def post(url, *, json, timeout):
            calls.append((url, json, timeout))
            return _Response()

    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)
    result = update_sglang_from_disk(
        "http://127.0.0.1:30000", tmp_path / "step.pt", 12, weight_version="12"
    )
    assert result == {"ok": True}
    assert calls == [
        (
            "http://127.0.0.1:30000/update_weights_from_disk",
            {
                "model_path": str(tmp_path / "step.pt"),
                "token_step": 12,
                "flush_cache": True,
                "abort_all_requests": True,
                "weight_version": "12",
            },
            600.0,
        )
    ]
