import html
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Literal


def clean_html_text(raw_html: str) -> str:
    """Helper to strip HTML tags, CDATA wrappers, and decode HTML entities."""
    text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', raw_html, flags=re.DOTALL)
    clean_text = re.sub(r'<[^<]+?>', '', text)
    return html.unescape(clean_text).strip()

def search_web_for_topic(query: str, max_results: int = 5) -> str:
    """
    Searches the live web for articles, news, discussions, and technical breakdowns about any topic.

    Args:
        query: The search query or technical topic to investigate.
        max_results: Maximum number of search results to return (default 5).
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/rss+xml, application/xml, text/xml, */*"
            }
        )
        
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_content = resp.read().decode("utf-8", errors="replace")

        items = re.findall(r"<item>(.*?)</item>", xml_content, re.DOTALL)
        if not items:
            fallback_query = query.split(":")[0].strip() if ":" in query else query
            if fallback_query != query:
                return search_web_for_topic(fallback_query, max_results=max_results)
            return f"No live search results found for query: '{query}'."

        results = []
        for idx, item in enumerate(items[:max_results], 1):
            title_match = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            pub_match = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
            link_match = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
            desc_match = re.search(r"<description>(.*?)</description>", item, re.DOTALL)

            title = clean_html_text(title_match.group(1)) if title_match else "Untitled"
            pub_date = pub_match.group(1).strip() if pub_match else "Recent"
            link = link_match.group(1).strip() if link_match else ""
            desc = clean_html_text(desc_match.group(1)) if desc_match else ""

            summary = (desc[:280] + "...") if len(desc) > 280 else desc

            results.append(
                f"[{idx}] {title}\n"
                f"    Date: {pub_date}\n"
                f"    Source/Link: {link}\n"
                f"    Summary: {summary}"
            )

        return "\n\n".join(results)
    except Exception as exc:
        return f"Live web search failed for '{query}': {exc!s}"


def fetch_trending_ai_topics(domain: Literal["ai", "cloud", "cybersecurity", "software_engineering", "general_tech"] = "ai") -> str:
    """
    Fetches real-time trending news, breakthroughs, and developments in AI and IT from live web feeds.

    Args:
        domain: Target tech domain to explore ('ai', 'cloud', 'cybersecurity', 'software_engineering', 'general_tech').
    """
    domain_query_map = {
        "ai": "Artificial Intelligence LLM AI agents breakthroughs",
        "cloud": "Cloud computing Kubernetes serverless infrastructure",
        "cybersecurity": "Cybersecurity zero trust security vulnerabilities AI",
        "software_engineering": "Software engineering developer tools architecture",
        "general_tech": "Emerging enterprise technology trends innovation"
    }
    
    query = domain_query_map.get(domain, "Artificial Intelligence latest news")
    return search_web_for_topic(query=query, max_results=6)


def search_topic_details(query: str, max_results: int = 5) -> str:
    """
    Searches the live web for in-depth facts, technical benchmarks, quotes, and metrics on a specific topic.

    Args:
        query: Specific technical query, product, or topic to research in detail.
        max_results: Maximum number of search results to return (default 5).
    """
    return search_web_for_topic(query=query, max_results=max_results)


def submit_for_human_review(post_content: str, review_notes: str, suggested_status: Literal["approved_for_human", "revision_needed"]) -> str:
    """
    Submits the reviewed LinkedIn post to the human reviewer queue.

    Args:
        post_content: The full text and formatting of the LinkedIn post.
        review_notes: Critique, suggestions, tone evaluation, and strengths identified by the reviewer agent.
        suggested_status: The reviewer agent's assessment ('approved_for_human' or 'revision_needed').
    """
    return (
        f"Post successfully staged for Human Review!\n"
        f"Status: {suggested_status}\n"
        f"Reviewer Notes: {review_notes}\n\n"
        f"--- Post Content ---\n{post_content}"
    )


def post_to_linkedin(post_content: str, access_token: str | None = None, author_urn: str | None = None, visibility: Literal["PUBLIC", "CONNECTIONS"] = "PUBLIC") -> str:
    """
    Publishes human-approved content directly to LinkedIn using the official LinkedIn UGC Posts API.
    Supports per-user OAuth2 access tokens and author URNs.

    Args:
        post_content: The finalized, human-approved text content of the LinkedIn post.
        access_token: The user-specific LinkedIn OAuth2 access token. If not provided, falls back to LINKEDIN_ACCESS_TOKEN env.
        author_urn: The user-specific LinkedIn Author URN (e.g. 'urn:li:person:12345678'). If not provided, fetched via /v2/me.
        visibility: Post visibility ('PUBLIC' or 'CONNECTIONS'). Default is 'PUBLIC'.
    """
    token = access_token or os.getenv("LINKEDIN_ACCESS_TOKEN")
    if not token:
        import time
        mock_id = f"urn:li:share:demo-{int(time.time())}"
        return f"Successfully published post to LinkedIn (Sandbox/Demo Mode)! Post ID: {mock_id}. (To publish live to real LinkedIn feed, provide LINKEDIN_ACCESS_TOKEN in .env)."

    resolved_author = author_urn or os.getenv("LINKEDIN_AUTHOR_URN")
    if not resolved_author:
        try:
            req_me = urllib.request.Request(
                "https://api.linkedin.com/v2/me",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }
            )
            with urllib.request.urlopen(req_me, timeout=10) as resp_me:
                me_data = json.loads(resp_me.read().decode("utf-8"))
                person_id = me_data.get("id")
                if person_id:
                    resolved_author = f"urn:li:person:{person_id}"
        except Exception as exc:
            return f"Error: Could not resolve LinkedIn Author URN from /v2/me using provided token ({exc!s}). Please provide author_urn explicitly."

    if not resolved_author:
        return "Error: Author URN could not be determined. Please specify author_urn."

    payload = {
        "author": resolved_author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": post_content
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": visibility
        }
    }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.linkedin.com/v2/ugcPosts",
            data=data_bytes,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0"
            },
            method="POST"
        )   

        with urllib.request.urlopen(req, timeout=15) as response:
            response_body = response.read().decode("utf-8")
            post_id = response.headers.get("x-restli-id") or response_body
            return f"Successfully published post to LinkedIn! Post URN/ID: {post_id}"

    except urllib.error.HTTPError as http_err:
        error_detail = http_err.read().decode("utf-8", errors="replace")
        return f"LinkedIn API HTTP Error {http_err.code}: {http_err.reason} - {error_detail}"
    except Exception as err:
        return f"Failed to post to LinkedIn due to unexpected error: {err!s}"
