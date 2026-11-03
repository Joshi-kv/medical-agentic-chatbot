from functools import partial
from typing import Any

from anyio import to_thread


class SupabaseDocumentStorage:
    def __init__(self, *, client: Any, bucket: str) -> None:
        self._bucket = client.storage.from_(bucket)

    async def upload(self, *, storage_key: str, content: bytes) -> None:
        await to_thread.run_sync(
            partial(
                self._bucket.upload,
                path=storage_key,
                file=content,
                file_options={
                    "content-type": "application/pdf",
                    "cache-control": "3600",
                    "upsert": "false",
                },
            )
        )

    async def delete(self, *, storage_key: str) -> None:
        await to_thread.run_sync(partial(self._bucket.remove, [storage_key]))
