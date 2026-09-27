"""Deterministic, locale-neutral normalization; never uses external data."""
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass

LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "llp", "plc", "pvt", "private", "sa", "sas", "gmbh"}
NAME_ABBR = {"corporation": "corp", "company": "co", "and": "and"}
ADDRESS_ABBR = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway", "apt": "apartment", "ste": "suite"}

def _ascii(value: object) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return value.lower().replace("&", " and ")

def _tokens(value: object, replacements: dict[str, str]) -> list[str]:
    text = re.sub(r"[^a-z0-9]+", " ", _ascii(value))
    return [replacements.get(t, t) for t in text.split()]

def normalize_name(value: object, remove_legal: bool = True) -> str:
    tokens = _tokens(value, NAME_ABBR)
    if remove_legal:
        tokens = [t for t in tokens if t not in LEGAL]
    return " ".join(tokens)

def normalize_address(value: object) -> str:
    return " ".join(_tokens(value, ADDRESS_ABBR))

def normalize_country(value: object) -> str:
    return " ".join(_tokens(value, {}))

def extract_postal(value: object) -> str:
    matches = re.findall(r"\b(?:\d{5}(?:-\d{4})?|\d{6}|[a-z]\d[a-z]\s?\d[a-z]\d)\b", _ascii(value))
    return matches[-1].replace(" ", "") if matches else ""

def extract_house_number(value: object) -> str:
    match = re.search(r"\b\d+[a-z]?\b", normalize_address(value))
    return match.group(0) if match else ""

@dataclass(frozen=True)
class AddressParts:
    normalized: str
    postal_code: str
    house_number: str

def address_parts(value: object) -> AddressParts:
    return AddressParts(normalize_address(value), extract_postal(value), extract_house_number(value))
