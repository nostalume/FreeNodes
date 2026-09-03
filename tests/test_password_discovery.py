import asyncio
from datetime import UTC, datetime

import pytest

from tests.discovery_support import (
    PASSWORD_ARTICLE,
    PASSWORD_DOWNLOAD,
    PASSWORD_ROOT,
    SIMPLE_ROOT,
    DiscoveryHarness,
    FakeDecryptionFactory,
    FakeLinkCapability,
    FakeWebCapability,
    FakeYouTubeCapability,
    article_link,
    config,
    page,
    password_source,
    subscription,
    success_capabilities,
    video_details,
    web_source,
    youtube_failure,
    youtube_source,
)
from tests.support import CapableProbe, ConsumerValidator, make_application


async def test_password_article_direct_link_bypasses_youtube_decryption_and_llm():
    site = password_source().model_copy(update={"resource_pattern": None})
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                html=f'<a href="{PASSWORD_DOWNLOAD}">direct</a>',
            ),
        },
        downloads={
            PASSWORD_DOWNLOAD: subscription("password.example", "direct"),
        },
    )
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    llm = FakeLinkCapability()
    outcome = await DiscoveryHarness(
        llm=llm, youtube=youtube, web=web, decryption=decryption
    ).run(site)

    assert outcome.kind == "success"
    assert youtube.inspected == []
    assert llm.calls == []
    assert decryption.clients[0].page_attempts == []


async def test_password_page_without_video_uses_bounded_policy():
    site = password_source()
    decrypted_url = "https://files.test/bruteforce.txt"
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                markdown="protected without video",
                html='<input type="password">',
            ),
        },
        downloads={decrypted_url: subscription("password.example", "bounded")},
    )
    decryption = FakeDecryptionFactory(
        page_results={
            "0011": page(PASSWORD_ARTICLE, markdown=decrypted_url),
        }
    )
    outcome = await DiscoveryHarness(web=web, decryption=decryption).run(site)

    attempted = decryption.clients[0].page_attempts[0][1]
    assert outcome.kind == "success"
    assert attempted == ("0011",)
    assert len(attempted) <= site.password_policy.max_candidates


async def test_unprotected_password_article_retains_reason():
    site = password_source()
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                markdown="no resource here",
                html="<p>ordinary article</p>",
            ),
        }
    )

    outcome = await DiscoveryHarness(web=web).run(site)

    assert outcome.kind == "failure"
    assert any("not password-protected" in error for error in outcome.errors)


async def test_password_discovery_scans_html_and_markdown_videos_in_stable_order():
    first = "https://youtube.com/watch?v=first123"
    second = "https://youtu.be/second456"
    site = password_source()
    web = FakeWebCapability(
        pages={
            PASSWORD_ROOT: page(PASSWORD_ROOT, links=article_link()),
            PASSWORD_ARTICLE: page(
                PASSWORD_ARTICLE,
                html=f'<input type="password"> terminal {first}',
                markdown=f"text begins here {first} then {second}",
            ),
        },
        downloads={
            PASSWORD_DOWNLOAD: subscription("password.example", "ordered"),
        },
    )
    youtube = FakeYouTubeCapability(
        details={
            first: youtube_failure(first, "first malformed"),
            second: video_details(second, subtitles="password 1122"),
        }
    )
    decryption = FakeDecryptionFactory(
        page_results={
            "1122": page(PASSWORD_ARTICLE, markdown=PASSWORD_DOWNLOAD),
        }
    )

    outcome = await DiscoveryHarness(
        youtube=youtube, web=web, decryption=decryption
    ).run(site)

    assert outcome.kind == "success"
    assert youtube.inspected == [first, second]


@pytest.mark.parametrize("variant", ("web", "password_page", "youtube_resources"))
async def test_variant_cancellation_propagates(variant):
    youtube = FakeYouTubeCapability()
    decryption = FakeDecryptionFactory()
    web = FakeWebCapability(pages={})
    if variant == "youtube_resources":
        site = youtube_source()
        youtube = FakeYouTubeCapability(cancel_listing=True)
    else:
        site = web_source() if variant == "web" else password_source()
        root = SIMPLE_ROOT if variant == "web" else PASSWORD_ROOT
        web = FakeWebCapability(pages={}, cancelled_urls={root})

    with pytest.raises(asyncio.CancelledError):
        await DiscoveryHarness(youtube=youtube, web=web, decryption=decryption).run(
            site
        )


async def test_all_variants_enter_one_deterministic_publication_flow(
    tmp_path,
):
    simple = web_source()
    password = password_source()
    drive_source = youtube_source()
    capabilities = success_capabilities()
    application = make_application(
        config(simple, password, drive_source),
        youtube_factory=capabilities.youtube,
        web_factory=capabilities.web,
        decryption_factory=capabilities.decryption,
        drive_factory=capabilities.drive,
    )
    application.llm = capabilities.llm
    receipt = await application.publish(
        repository_root=tmp_path,
        validator=ConsumerValidator(
            required_files=("nodes/merged.yaml", "nodes/provider.yaml")
        ),
        probe_session=CapableProbe(),
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert receipt.status == "accepted"
    assert capabilities.youtube.factory_calls == [("", 3)]
    assert capabilities.web.factory_calls == 1
    assert capabilities.decryption.calls == [("", 30.0), ("", 30.0)]
    assert all(client.enter_calls == 1 for client in capabilities.decryption.clients)
    assert all(client.close_calls == 1 for client in capabilities.decryption.clients)
    assert capabilities.drive.calls == [("", 30.0)]
    assert capabilities.drive.clients[0].enter_calls == 1
    assert capabilities.drive.clients[0].close_calls == 1
    assert (tmp_path / "nodes" / "publication-receipt.json").exists()
