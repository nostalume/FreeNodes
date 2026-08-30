"""Opt-in real Mihomo process and endpoint smoke."""

import os
from pathlib import Path

import pytest

from src.mihomo import MihomoDelayProbe, MihomoProbeSession
from src.nodes import ClashNode, admit_proxy


@pytest.mark.skipif(
    not os.environ.get("MIHOMO_REAL_PROBE"), reason="real Mihomo path not supplied"
)
async def test_real_direct_succeeds_and_closed_socks_port_fails():
    executable = Path(os.environ["MIHOMO_REAL_PROBE"])
    direct = ClashNode(
        fingerprint="d" * 64,
        display_name="DIRECT-FIXTURE",
        proxy=admit_proxy({"name": "DIRECT-FIXTURE", "type": "direct", "udp": True}),
    )
    unreachable = ClashNode(
        fingerprint="e" * 64,
        display_name="UNREACHABLE-FIXTURE",
        proxy=admit_proxy(
            {
                "name": "UNREACHABLE-FIXTURE",
                "type": "socks5",
                "server": "127.0.0.1",
                "port": 9,
            }
        ),
    )
    session = MihomoProbeSession(
        executable,
        delay_probe=MihomoDelayProbe(timeout_ms=2500, deadline=20),
    )

    evidence = await session.probe((direct, unreachable))

    assert evidence[0].status == "success"
    assert evidence[0].coarse.delay_ms is not None
    assert evidence[0].confirm is not None and evidence[0].confirm.delay_ms is not None
    assert evidence[1].status in {"timeout", "api_error"}
