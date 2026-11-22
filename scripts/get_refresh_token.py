"""Reconnect the configured Gmail account and save its refresh token privately."""
import argparse
from getpass import getpass
from pathlib import Path
import os
import sys
from secrets import compare_digest
from urllib.parse import parse_qs, urlsplit
import webbrowser

from dotenv import dotenv_values, set_key
from google_auth_oauthlib.flow import Flow, InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def hosted_code(callback: str, redirect: str, state: str) -> str:
    expected = urlsplit(redirect)
    actual = urlsplit(callback)
    if ((expected.scheme, expected.netloc, expected.path) != (actual.scheme, actual.netloc, actual.path)
            or actual.fragment):
        raise ValueError("Callback URL does not match GOOGLE_REDIRECT_URI")
    query = parse_qs(actual.query)
    states, codes = query.get("state", []), query.get("code", [])
    if len(states) != 1 or not compare_digest(states[0].encode(), state.encode()):
        raise ValueError("OAuth state did not match")
    if len(codes) != 1 or "error" in query:
        raise ValueError("Google did not return an authorization code")
    return codes[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosted", action="store_true", help="Use the configured HTTPS callback")
    args = parser.parse_args()
    env_path = Path(__file__).resolve().parents[1] / ".env"
    values = dotenv_values(env_path)
    client_id = os.getenv("GOOGLE_CLIENT_ID") or values.get("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or values.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        parser.error("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the project .env first")

    client = {
        "client_id": client_id,
        "client_secret": client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    try:
        if args.hosted:
            if not sys.stdin.isatty() or not sys.stderr.isatty():
                parser.error("Run hosted authorization from an interactive terminal")
            redirect = os.getenv("GOOGLE_REDIRECT_URI") or values.get("GOOGLE_REDIRECT_URI", "")
            url = urlsplit(redirect or "")
            if (url.scheme != "https" or not url.hostname or url.username or url.password
                    or url.path != "/oauth/gmail/callback" or url.query or url.fragment):
                parser.error("Set GOOGLE_REDIRECT_URI to your HTTPS /oauth/gmail/callback URL")
            flow = Flow.from_client_config({"web": client}, SCOPES, autogenerate_code_verifier=True)
            flow.redirect_uri = redirect
            auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
            if not webbrowser.open(auth_url):
                parser.error("Open hosted authorization from a machine with a local web browser")
            callback = getpass("Paste the redirected browser address here (hidden): ")
            flow.fetch_token(code=hosted_code(callback.strip(), redirect, state))
            creds = flow.credentials
        else:
            flow = InstalledAppFlow.from_client_config(
                {"installed": client}, SCOPES, autogenerate_code_verifier=True,
            )
            creds = flow.run_local_server(
                port=0, access_type="offline", prompt="consent", timeout_seconds=180,
                authorization_prompt_message=None,
            )
    except Exception as exc:
        parser.error(f"Google authorization could not complete ({type(exc).__name__})")
    if not creds.refresh_token:
        parser.error("Google did not return a refresh token. Authorization was not saved.")
    env_path.touch(mode=0o600, exist_ok=True)
    os.chmod(env_path, 0o600)
    set_key(str(env_path), "GOOGLE_REFRESH_TOKEN", creds.refresh_token, quote_mode="never")
    os.chmod(env_path, 0o600)
    print("Gmail authorization saved to the project .env. Update the host secret or restart Pingo.")


if __name__ == "__main__":
    main()
