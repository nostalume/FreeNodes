"""Opt-in pinned-Mihomo process integration against disposable origins."""

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from src.nodes import ClashNode, admit_proxy
from src.probe import MihomoDelayProbe, MihomoProbeSession, TransferTarget
from src.quality import ProbePlan, ProbePlanEntry, QualityPolicy


@pytest.mark.skipif(
    not os.environ.get("MIHOMO_REAL_PROBE"), reason="real Mihomo path not supplied"
)
async def test_real_process_probes_serial_blocks_and_releases_local_origins():
    async def reject_socks(reader, writer):
        writer.close()
        await writer.wait_closed()

    async def serve_transfer(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 1048576\r\n"
                b"Connection: close\r\n\r\n" + b"x" * 1048576
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    socks_server = await asyncio.start_server(reject_socks, "127.0.0.1", 0)
    transfer_server = await asyncio.start_server(serve_transfer, "127.0.0.1", 0)
    socks_port = socks_server.sockets[0].getsockname()[1]
    transfer_port = transfer_server.sockets[0].getsockname()[1]
    targets = tuple(
        TransferTarget(
            name=f"fixture-{name}",
            authority=f"fixture-{name}",
            url=f"http://127.0.0.1:{transfer_port}/exact-1mib",
        )
        for name in ("a", "b")
    )
    direct = tuple(
        ClashNode(
            fingerprint=f"{index:064x}",
            display_name=f"DIRECT-FIXTURE-{index}",
            proxy=admit_proxy(
                {"name": f"DIRECT-FIXTURE-{index}", "type": "direct", "udp": True}
            ),
        )
        for index in range(1, 17)
    )
    unreachable = ClashNode(
        fingerprint="e" * 64,
        display_name="UNREACHABLE-FIXTURE",
        proxy=admit_proxy(
            {
                "name": "UNREACHABLE-FIXTURE",
                "type": "socks5",
                "server": "127.0.0.1",
                "port": socks_port,
            }
        ),
    )

    async def admit_delays(url: str, timeout: float):
        if url.endswith("/version"):
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()
                return response.json()
        return {"delay": 20}

    nodes = (*direct, unreachable)
    plan = ProbePlan(
        entries=tuple(
            ProbePlanEntry(
                ordinal=index,
                node=node,
                sources=("fixture",),
                protocol=node.proxy.type,
            )
            for index, node in enumerate(nodes)
        )
    )
    session = MihomoProbeSession(
        Path(os.environ["MIHOMO_REAL_PROBE"]),
        delay_probe=MihomoDelayProbe(request_json=admit_delays),
        transfer_targets=targets,
        transfer_timeout=2.0,
    )

    try:
        result = await session.probe(plan, QualityPolicy())
    finally:
        socks_server.close()
        transfer_server.close()
        await socks_server.wait_closed()
        await transfer_server.wait_closed()

    assert result.status == "success"
    assert all(item.transfer.status == "success" for item in result.evidence[:16])
    assert result.evidence[16].transfer.status != "success"
    assert [target.candidate_attempts for target in result.transfer_targets] == [9, 9]
    assert [target.controls_passed for target in result.transfer_targets] == [4, 4]
