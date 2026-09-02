"""Opt-in pinned-Mihomo integration with reviewed HTTPS controls."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.nodes import ClashNode, NodeProvenance, admit_proxy
from src.probe import MihomoProbeSession
from src.quality import (
    DEFAULT_CAPABILITY_TARGETS,
    ProbePlan,
    ProbePlanEntry,
    QualityPolicy,
)

NOW = datetime(2026, 9, 2, tzinfo=UTC)


@pytest.mark.skipif(
    not os.environ.get("MIHOMO_REAL_PROBE"), reason="real Mihomo path not supplied"
)
async def test_real_process_proves_capability_and_releases_local_proxy():
    async def respond(reader, writer):
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            if b"raw.githubusercontent.com" in request:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
                )
            else:
                writer.write(
                    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    proxy = await asyncio.start_server(respond, "127.0.0.1", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]
    node = ClashNode(
        fingerprint=f"{1:064x}",
        display_name="FIXTURE",
        proxy=admit_proxy(
            {
                "name": "FIXTURE",
                "type": "http",
                "server": "127.0.0.1",
                "port": proxy_port,
            }
        ),
        provenance=(
            NodeProvenance(
                authority="fixture",
                site="fixture",
                source_url="https://fixture.test/nodes",
                observed_at=NOW,
                artifact_digest=f"{11:064x}",
                item_index=1,
            ),
        ),
    )
    plan = ProbePlan(
        entries=(
            ProbePlanEntry(
                ordinal=0,
                node=node,
                sources=("fixture",),
                protocol=node.proxy.type,
            ),
        )
    )

    try:
        result = await MihomoProbeSession(
            Path(os.environ["MIHOMO_REAL_PROBE"])
        ).probe_capabilities(plan, DEFAULT_CAPABILITY_TARGETS, QualityPolicy())
    finally:
        proxy.close()
        await proxy.wait_closed()

    assert result.status == "complete", result.model_dump()
    assert result.accepted_fingerprints == (node.fingerprint,), result.model_dump()
    assert result.decisions[0].status == "capable"
