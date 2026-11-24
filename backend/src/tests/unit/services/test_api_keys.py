import pytest

from services.api_keys import (
    PATIENT_SCOPES,
    APIKeyRecord,
    APIKeyService,
    InvalidAPIKeyInput,
)


class MemoryAPIKeyStore:
    def __init__(self) -> None:
        self.records: dict[str, APIKeyRecord] = {}

    def create(self, record: APIKeyRecord) -> APIKeyRecord:
        self.records[record.key_hash] = record
        return record

    def find_active(self, key_hash: str) -> APIKeyRecord | None:
        return self.records.get(key_hash)


def test_issue_returns_plaintext_once_and_stores_only_hash() -> None:
    store = MemoryAPIKeyStore()
    service = APIKeyService(store)

    issued = service.issue("  Maya  ", "tenant-1")

    assert issued.api_key.startswith("med_")
    assert issued.record.name == "Maya"
    assert issued.record.scopes == PATIENT_SCOPES
    assert issued.record.key_hash == service.hash_key(issued.api_key)
    assert issued.api_key not in repr(store.records)
    assert service.authenticate(issued.api_key) == issued.record


@pytest.mark.parametrize("name", ["", " ", "A", "x" * 101])
def test_issue_rejects_invalid_name(name: str) -> None:
    with pytest.raises(InvalidAPIKeyInput):
        APIKeyService(MemoryAPIKeyStore()).issue(name, "tenant-1")


def test_authenticate_unknown_key_returns_none() -> None:
    assert APIKeyService(MemoryAPIKeyStore()).authenticate("med_unknown") is None


def test_hash_is_stable() -> None:
    assert APIKeyService.hash_key("med_example") == APIKeyService.hash_key(
        "med_example"
    )
    assert APIKeyService.hash_key("med_example") != "med_example"
