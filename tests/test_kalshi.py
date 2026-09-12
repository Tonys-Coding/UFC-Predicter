import base64
from copy import deepcopy
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from database import utc_now
from kalshi_mma_client import (
    KalshiError,
    KalshiMMAClient,
    is_mma_series,
    is_winner_series,
    parse_event_markets,
    price_probability,
)


def public_client(tmp_path):
    return KalshiMMAClient(api_key="", private_key_path="", passphrase="", cache_dir=tmp_path)


def test_dollars_legacy_cents_and_invalid_quotes():
    assert price_probability({"yes_ask": 65}, "yes_ask") == 0.65
    assert price_probability({"yes_ask_dollars": "0.6510", "yes_ask": 60}, "yes_ask") == 0.651
    for raw in ["nan", "inf", "1.2", "-0.1", "bad", True]:
        assert price_probability({"yes_ask_dollars": raw}, "yes_ask") is None


def test_pairing_uses_other_yes_contract_not_no_subtitle(event):
    rows = parse_event_markets(event, utc_now())
    assert len(rows) == 2 and rows[0]["opponent_name"] == "Blake Test"
    assert rows[0]["kalshi_probability"] == 0.6
    del event["markets"][0]["yes_ask_dollars"]
    event["markets"][0]["no_bid_dollars"] = ".39"
    assert parse_event_markets(event, utc_now())[0]["kalshi_probability"] == pytest.approx(0.61)


@pytest.mark.parametrize(
    "field,value",
    [
        ("yes_ask_dollars", "0"),
        ("yes_ask_dollars", "1"),
        ("status", "closed"),
        ("occurrence_datetime", "2020-01-01T00:00:00Z"),
        ("close_time", "2020-01-01T00:00:00Z"),
        ("yes_ask_size_fp", "0"),
        ("yes_bid_dollars", ".99"),
    ],
)
def test_non_executable_and_started_markets_filtered(event, field, value):
    event["markets"][0][field] = value
    rows = parse_event_markets(event, utc_now())
    assert all(row["fighter_name"] != "Alex Test" for row in rows)


def test_props_nonexclusive_events_and_false_mma_matches_rejected(event):
    for ticker in ["KXSIXKINGSSLAMMATCH", "KXRUGBYGPREMMATCH"]:
        assert not is_mma_series({"ticker": ticker})
    assert is_winner_series({"ticker": "KXUFCFIGHT", "title": "UFC Fight"})
    assert not is_winner_series({"ticker": "KXUFCMOV", "title": "UFC Method of Victory"})
    event["markets"][0]["title"] = "Alex Test wins by KO"
    assert not parse_event_markets(event, utc_now())
    event["mutually_exclusive"] = False
    assert not parse_event_markets(event, utc_now())


def test_pagination_repeated_cursor_fails_instead_of_silent_partial_data(tmp_path):
    client = public_client(tmp_path)
    client._get = Mock(return_value={"events": [{"event_ticker": "a"}], "cursor": "same"})
    with pytest.raises(KalshiError, match="repeated"):
        list(client._pages("/events", "events", {}))


def test_live_catalog_pagination_and_cache(event, tmp_path):
    client = public_client(tmp_path)
    last = deepcopy(event)
    last["event_ticker"] += "2"
    for m in last["markets"]:
        m["ticker"] += "2"
    client._get = Mock(
        side_effect=[
            {"series": [{"ticker": "KXUFCFIGHT", "title": "UFC Fight"}]},
            {
                "milestones": [
                    {
                        "start_date": event["markets"][0]["occurrence_datetime"],
                        "details": {"league": "UFC", "status": "not_started"},
                        "related_event_tickers": [event["event_ticker"], last["event_ticker"]],
                    }
                ]
            },
            {"events": [event], "cursor": "next"},
            {"events": [last], "cursor": ""},
        ]
    )
    data = client.get_upcoming_ufc_markets()
    assert len(data) == 4 and not data.attrs["stale"]
    cached = client.load_cached_markets()
    assert len(cached) == 4 and cached.attrs["stale"]
    assert client._get.call_args_list[-1].args[1]["cursor"] == "next"


def test_live_fight_rejected_even_when_market_expires_in_future(event):
    milestone = {
        "start_date": event["markets"][0]["occurrence_datetime"],
        "details": {"status": "live"},
    }
    assert parse_event_markets(event, utc_now(), milestone=milestone) == []
    milestone["details"]["status"] = "not_started"
    milestone["start_date"] = "2020-01-01T00:00:00Z"
    assert parse_event_markets(event, utc_now(), milestone=milestone) == []


def test_settlement_time_is_never_used_as_fight_start(event):
    for market in event["markets"]:
        market["expected_expiration_time"] = market.pop("occurrence_datetime")
    assert parse_event_markets(event, utc_now()) == []


def test_rsa_pss_signature_excludes_query_string(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "test-key.pem"
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"test-only"),
        )
    )
    client = KalshiMMAClient(
        api_key="test-key-id",
        private_key_path=str(path),
        passphrase="test-only",
        cache_dir=tmp_path,
    )
    headers = client._headers("GET", "/trade-api/v2/markets?limit=2")
    message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET/trade-api/v2/markets").encode()
    key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_credentials_never_sent_to_untrusted_origin(tmp_path):
    with pytest.raises(KalshiError, match="official HTTPS"):
        KalshiMMAClient(
            base_url="https://example.com/trade-api/v2", api_key="secret", cache_dir=tmp_path
        )
    with pytest.raises(KalshiError, match="require"):
        KalshiMMAClient(api_key="id", private_key_path="", passphrase="", cache_dir=tmp_path)
