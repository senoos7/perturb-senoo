"""Cloudflare R2 (S3-compatible) upload helpers for miner challenge images."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    prefix: str = "challenges/"
    endpoint_url: str = ""

    @classmethod
    def from_env(cls) -> R2Config:
        from perturbnet import constants as C

        account_id = C.R2_ACCOUNT_ID
        access_key_id = C.R2_ACCESS_KEY_ID
        secret_access_key = C.R2_SECRET_ACCESS_KEY
        if not account_id or not access_key_id or not secret_access_key:
            raise ValueError(
                "R2 upload requires PERTURB_R2_ACCOUNT_ID, PERTURB_R2_ACCESS_KEY_ID, "
                "and PERTURB_R2_SECRET_ACCESS_KEY"
            )
        endpoint_url = C.R2_ENDPOINT or f"https://{account_id}.r2.cloudflarestorage.com"
        prefix = C.R2_PREFIX or "challenges/"
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        bucket = C.R2_BUCKET or "perturb-miner-challenge"
        return cls(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
        )


class ChallengeR2Uploader:
    def __init__(self, config: R2Config) -> None:
        self._config = config
        self._client: Any | None = None

    @classmethod
    def from_env(cls) -> ChallengeR2Uploader:
        return cls(R2Config.from_env())

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self._config.endpoint_url,
                aws_access_key_id=self._config.access_key_id,
                aws_secret_access_key=self._config.secret_access_key,
                region_name="auto",
            )
        return self._client

    def upload_jpeg(
        self,
        filename: str,
        data: bytes,
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        key = f"{self._config.prefix}{filename.lstrip('/')}"
        put_kwargs: dict[str, Any] = {
            "Bucket": self._config.bucket,
            "Key": key,
            "Body": data,
            "ContentType": "image/jpeg",
        }
        if metadata:
            put_kwargs["Metadata"] = {k: str(v)[:256] for k, v in metadata.items()}
        self._get_client().put_object(**put_kwargs)
        return key

    @property
    def bucket(self) -> str:
        return self._config.bucket

    @property
    def prefix(self) -> str:
        return self._config.prefix
