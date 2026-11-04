from __future__ import annotations

import builtins
import os
import sys
from types import SimpleNamespace

from cap_harness.providers.wevisone import service


def test_local_service_sets_offline_before_importing_model(tmp_path, monkeypatch) -> None:
    real_import = builtins.__import__
    imported_offline = []

    def import_model(name, global_scope=None, local_scope=None, fromlist=(), level=0):
        if name == "wevisone_vl.engine":
            imported_offline.append(
                (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
            )
            return SimpleNamespace(WeVisOneVL=lambda *_args, **_kwargs: object())
        return real_import(name, global_scope, local_scope, fromlist, level)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_model)
    monkeypatch.setattr(sys, "argv", ["service", "--model", str(tmp_path)])
    monkeypatch.setattr(service, "verify_snapshot", lambda _path: None)
    monkeypatch.setattr(service.uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_ENGINE", None)

    service.main()

    assert imported_offline == [("1", "1")]
