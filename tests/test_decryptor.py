"""Admitted page recognition used by browser decryption."""

import pytest

from src.crawler import Page
from src.decryptor import extract_paste_links, extract_paste_url


def page(*, markdown: str = "", html: str = "") -> Page:
    return Page(url="https://fixture.test", markdown=markdown, html=html)


@pytest.mark.parametrize(
    "html",
    (
        '<input class="cl-input" placeholder="在此输入密码">',
        '<button class="cl-btn">解密</button>',
        '<input type="password" name="pwd">',
        '<input type="Password" name="key">',
    ),
)
def test_page_recognizes_structural_password_controls(html):
    assert page(html=html).requires_password() is True


@pytest.mark.parametrize(
    "html",
    (
        "<div>请输入密码查看内容</div>",
        "<div>clash免费节点</div>",
        "<div>ordinary page</div>",
    ),
)
def test_page_does_not_infer_protection_from_text(html):
    assert page(html=html).requires_password() is False


@pytest.mark.parametrize(
    ("markdown", "html"),
    (
        ("", "https://example.com/node.txt"),
        ("", "https://example.com/config.yaml"),
        ("vmess://eyJ2IjoiMiI6ICJhYmNk", ""),
        ("vless://abc@1.2.3.4:443", ""),
        ("trojan://pass@1.2.3.4:443", ""),
        ("ss://YWVzLTI1Ni1nY206d2MvZXFSUHJZ", ""),
    ),
)
def test_page_recognizes_subscription_content(markdown, html):
    assert page(markdown=markdown, html=html).has_subscription_content() is True


@pytest.mark.parametrize(
    ("markdown", "html"),
    (
        ("Clash 订阅配置文件", ""),
        ("", "<html>普通网页内容</html>"),
        ("", ""),
    ),
)
def test_page_rejects_descriptive_or_empty_content(markdown, html):
    assert page(markdown=markdown, html=html).has_subscription_content() is False


class TestExtractPasteUrl:
    def test_paste_to_with_fragment(self):
        text = "资源地址：https://paste.to/?3c4d47bd5fa1f66a#BwJn7AXEmdXR88rdRZyY7JXjKrmd8NgcjVwU2SiwroVf"
        result = extract_paste_url(text)
        assert result is not None
        assert "#BwJn" in result
        assert "3c4d47bd5fa1f66a" in result

    def test_youtube_redirect_with_paste_url(self):
        text = (
            "https://www.youtube.com/redirect?event=video_description"
            "&redir_token=fixture"
            "&q=https%3A%2F%2Fpaste.to%2F%3Ffixture%23secret"
            "&v=fixture"
        )
        result = extract_paste_url(text)
        assert result is not None
        assert "paste.to" in result

    def test_no_paste_url(self):
        assert extract_paste_url("https://example.com") is None

    def test_paste_without_fragment_returns_none(self):
        assert extract_paste_url("https://paste.to/?fixture") is None

    def test_privatebin_url(self):
        result = extract_paste_url(
            "https://privatebin.example.com/?abc123#secretkey456"
        )
        assert result is not None
        assert "secretkey456" in result


class TestExtractPasteLinks:
    @pytest.mark.parametrize(
        "url",
        (
            "https://example.com/v2ray.txt",
            "https://example.com/config.yaml",
            "https://dlink.host/1drv/encoded.jpg",
            "https://1drv.ms/f/c/fixture.jpg",
        ),
    )
    def test_subscription_link(self, url):
        links = extract_paste_links(f"subscription: {url}")
        assert tuple(link.href for link in links) == (url,)

    def test_regular_image_is_ignored(self):
        assert extract_paste_links("https://example.com/photo.jpg") == ()

    def test_empty_text(self):
        assert extract_paste_links("") == ()

    def test_multiple_links(self):
        links = extract_paste_links(
            "v2ray: https://example.com/v2.txt\n"
            "clash: https://example.com/c.yaml\n"
            "onedrive: https://dlink.host/1drv/abc.jpg"
        )
        assert len(links) == 3
