# SPDX-License-Identifier: Apache-2.0
# Standard
import dataclasses

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_list,
    apply_mm_hashes_to_token_ids,
    hex_hash_to_int16,
    hex_hash_to_int64,
    get_mm_aware_token_ids,
)


@dataclasses.dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


@dataclasses.dataclass(frozen=True)
class DummyMMFeature:
    identifier: str
    mm_position: DummyPlaceholderRange


@dataclasses.dataclass
class DummyRequest:
    all_token_ids: list[int]
    mm_features: list[DummyMMFeature] | None = None
    mm_hashes: list[str] | None = None
    mm_positions: list[DummyPlaceholderRange] | None = None


def test_hex_hash_to_int64_accepts_hex_and_non_hex() -> None:
    assert hex_hash_to_int64("0000") == 0
    assert hex_hash_to_int64("0x0001") == 1
    assert hex_hash_to_int64("0x10000") > 0xFFFF
    assert hex_hash_to_int64("0xffffffffffffffff") == (2**63 - 1)

    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int64(s)
    v2 = hex_hash_to_int64(s)
    assert isinstance(v1, int)
    assert 0 <= v1 < 2**63
    assert v1 == v2


def test_hex_hash_to_int16_accepts_hex_and_non_hex() -> None:
    # Hex behavior preserved (with and without 0x prefix).
    assert hex_hash_to_int16("0000") == 0
    assert hex_hash_to_int16("ffff") == 0xFFFF
    assert hex_hash_to_int16("0xFFFF") == 0xFFFF
    assert hex_hash_to_int16("0x0001") == 1

    # Non-hex identifiers must not raise and must be deterministic.
    s = "chatcmpl-a2a48871c4aad192-image-0"
    v1 = hex_hash_to_int16(s)
    v2 = hex_hash_to_int16(s)
    assert isinstance(v1, int)
    assert 0 <= v1 <= 0xFFFF
    assert v1 == v2
    assert v1 == (hex_hash_to_int64(s) & 0xFFFF)


def test_hex_hash_to_int16_hex_variants_whitespace_and_truncation() -> None:
    # Whitespace should be ignored and case should not matter.
    assert hex_hash_to_int16(" FfFf ") == 0xFFFF
    assert hex_hash_to_int16("\n0x00aB\t") == 0x00AB

    # Long hex should be truncated to 16 bits via masking.
    assert hex_hash_to_int16("123456") == 0x3456
    assert hex_hash_to_int16("0x123456") == 0x3456


def test_hex_hash_to_int16_empty_and_invalid_hex_are_safe_and_deterministic() -> None:
    # Empty (or effectively empty) values should not raise.
    for s in ("", "   ", "0x"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2

    # Invalid "hex-looking" strings must fall back to hashing.
    for s in ("0xGG", "deadbeeg", "0x12xz"):
        v1 = hex_hash_to_int16(s)
        v2 = hex_hash_to_int16(s)
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2


def test_hex_hash_to_int16_non_string_inputs_are_safe() -> None:
    # Be defensive: callers may pass None or other non-string types.
    for val in (None, 0, 12345, 3.14, b"deadbeef"):
        v1 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        v2 = hex_hash_to_int16(val)  # type: ignore[arg-type]
        assert isinstance(v1, int)
        assert 0 <= v1 <= 0xFFFF
        assert v1 == v2
        assert v1 == (hex_hash_to_int64(val) & 0xFFFF)


def test_apply_mm_hashes_to_token_ids_handles_non_hex_mm_hash() -> None:
    token_ids = torch.arange(0, 10, dtype=torch.long)
    mm_hashes = ["chatcmpl-a2a48871c4aad192-image-0"]
    mm_positions = [DummyPlaceholderRange(offset=2, length=4)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    expected_val = hex_hash_to_int64(mm_hashes[0])
    assert out[2:6].tolist() == [expected_val] * 4
    assert expected_val > 0xFFFF


def test_apply_mm_hashes_to_token_ids_out_of_bounds_is_safe() -> None:
    token_ids = torch.zeros(5, dtype=torch.long)
    mm_hashes = ["deadbeef"]
    mm_positions = [DummyPlaceholderRange(offset=999, length=10)]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    assert out.tolist() == token_ids.tolist()


def test_apply_mm_hashes_to_token_ids_multiple_placeholders_and_length_mismatch() -> (
    None
):
    token_ids = torch.zeros(12, dtype=torch.long)
    mm_hashes = ["deadbeef", "chatcmpl-a2a48871c4aad192-image-0", "EXTRA_HASH_IGNORED"]
    mm_positions = [
        DummyPlaceholderRange(offset=0, length=3),
        DummyPlaceholderRange(offset=5, length=4),
    ]

    out = apply_mm_hashes_to_token_ids(token_ids.clone(), mm_hashes, mm_positions)
    v0 = hex_hash_to_int64(mm_hashes[0])
    v1 = hex_hash_to_int64(mm_hashes[1])

    assert out[0:3].tolist() == [v0] * 3
    assert out[5:9].tolist() == [v1] * 4
    # Other regions remain unchanged.
    assert out[3:5].tolist() == [0, 0]
    assert out[9:12].tolist() == [0, 0, 0]


def test_apply_mm_hashes_to_token_list_returns_rewritten_copy() -> None:
    token_ids = [10, 11, 12, 13, 14]
    mm_hashes = ["0x10000"]
    mm_positions = [DummyPlaceholderRange(offset=1, length=3)]

    out = apply_mm_hashes_to_token_list(token_ids, mm_hashes, mm_positions)

    assert token_ids == [10, 11, 12, 13, 14]
    assert out == [10, 0x10000, 0x10000, 0x10000, 14]


def test_get_mm_aware_token_ids_text_only_returns_copy() -> None:
    request = DummyRequest(all_token_ids=[1, 2, 3])

    out = get_mm_aware_token_ids(request)

    assert out == [1, 2, 3]
    assert out is not request.all_token_ids


def test_get_mm_aware_token_ids_uses_mm_features() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4, 5],
        mm_features=[
            DummyMMFeature(
                identifier="0x10000",
                mm_position=DummyPlaceholderRange(offset=2, length=2),
            )
        ],
    )

    assert get_mm_aware_token_ids(request) == [1, 2, 0x10000, 0x10000, 5]


def test_get_mm_aware_token_ids_uses_legacy_fields() -> None:
    request = DummyRequest(
        all_token_ids=[1, 2, 3, 4, 5, 6],
        mm_hashes=["image-a", "image-b"],
        mm_positions=[
            DummyPlaceholderRange(offset=1, length=2),
            DummyPlaceholderRange(offset=4, length=5),
        ],
    )

    out = get_mm_aware_token_ids(request)

    image_a = hex_hash_to_int64("image-a")
    image_b = hex_hash_to_int64("image-b")
    assert out == [1, image_a, image_a, 4, image_b, image_b]
    assert request.all_token_ids == [1, 2, 3, 4, 5, 6]
