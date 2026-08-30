"""Deterministic subscription-link parsing before LLM fallback."""

from src.llm_router import ExtractedLinks


def test_deterministic_parser_classifies_and_stable_deduplicates():
    parsed = ExtractedLinks.from_text(
        "\n".join(
            (
                "https://files.test/nodes.txt",
                "https://files.test/config.yaml?token=one",
                "vmess://eyJ2IjoiMiJ9",
                "https://files.test/nodes.txt",
            )
        )
    )

    assert parsed.txt == ("https://files.test/nodes.txt",)
    assert parsed.yaml == ("https://files.test/config.yaml?token=one",)
    assert parsed.inline == ("vmess://eyJ2IjoiMiJ9",)


def test_deterministic_parser_rejects_descriptive_text():
    assert ExtractedLinks.from_text("Clash and V2Ray configuration") == ExtractedLinks()
