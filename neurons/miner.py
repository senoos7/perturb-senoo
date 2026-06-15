import argparse
import concurrent.futures
import json
import logging as pylogging
import os
import time
import typing
from pathlib import Path

import bittensor as bt
import torch

from perturbnet.attacks import list_attack_algorithms, run_attack, runtime_profile_summary
from perturbnet.constants import (
    CHALLENGE_SAVE_DIR,
    MINER_VALIDATOR_IP_WHITELIST,
    R2_UPLOAD_CHALLENGES,
    SAVE_CHALLENGE_IMAGES,
)
from perturbnet.image_io import (
    challenge_image_b64_to_jpeg_bytes,
    decode_image_b64,
    encode_image_b64,
    sanitize_task_id_for_filename,
)
from perturbnet.model import load_efficientnet_v2_l, resolve_target_index
from perturbnet.protocol import AttackChallenge
from perturbnet.r2_storage import ChallengeR2Uploader

logger = pylogging.getLogger(__name__)

# Disable with MINER_VALIDATOR_IP_WHITELIST=off
_IP_WHITELIST_DISABLED = frozenset({"off", "false", "0", "no", "*", "all"})


def _parse_validator_ip_whitelist(raw: str) -> list[tuple[str, int | None]]:
    """Parse comma-separated entries: ``ip`` or ``ip:axon_port``."""
    entries: list[tuple[str, int | None]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            ip, port_raw = item.rsplit(":", 1)
            entries.append((ip.strip(), int(port_raw.strip())))
        else:
            entries.append((item, None))
    return entries


def _caller_ip_whitelisted(
    *,
    caller_ip: str | None,
    caller_uid: int | None,
    metagraph: typing.Any,
    whitelist: list[tuple[str, int | None]],
) -> tuple[bool, str]:
    if not caller_ip:
        return False, "Missing caller IP"
    for allowed_ip, allowed_axon_port in whitelist:
        if caller_ip != allowed_ip:
            continue
        if allowed_axon_port is None:
            return True, "OK"
        if caller_uid is None:
            return False, f"Caller IP {caller_ip} requires registered axon port {allowed_axon_port}"
        axon = metagraph.axons[caller_uid]
        axon_ip = str(getattr(axon, "ip", "") or "").strip()
        axon_port = int(getattr(axon, "port", 0) or 0)
        if axon_ip == allowed_ip and axon_port == allowed_axon_port:
            return True, "OK"
        return (
            False,
            f"Caller axon {axon_ip}:{axon_port} does not match whitelist {allowed_ip}:{allowed_axon_port}",
        )
    return False, f"Caller IP {caller_ip} not in whitelist"


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _resolve_axon_port(config: typing.Any) -> int:
    """bt.config() defaults axon.port to 8091; prefer MINER_PORT / CLI override."""
    for raw in (
        os.getenv("MINER_PORT"),
        os.getenv("AXON_PORT"),
        getattr(config, "axon_port", None),
    ):
        if raw is not None and str(raw).strip():
            return int(raw)
    axon_port = getattr(getattr(config, "axon", None), "port", None)
    if axon_port is not None:
        return int(axon_port)
    return 9000


def _make_axon(wallet, port: int):
    resolved_port = int(port)
    if hasattr(bt, "axon"):
        return bt.axon(wallet=wallet, port=resolved_port)
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    return axon_cls(wallet=wallet, port=resolved_port)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_v2_l(self.device)
        logger.info(f"[MINER] Attack profile: {runtime_profile_summary()}")

        # Validator IP whitelist — disabled until real validator endpoint is confirmed.
        # Re-enable by uncommenting below and setting MINER_VALIDATOR_IP_WHITELIST, e.g. 63.142.193.28:31711
        self._validator_ip_whitelist: list[tuple[str, int | None]] = []
        # whitelist_raw = MINER_VALIDATOR_IP_WHITELIST
        # if whitelist_raw.lower() in _IP_WHITELIST_DISABLED:
        #     self._validator_ip_whitelist = []
        #     logger.warning("[MINER] Validator IP whitelist disabled")
        # else:
        #     self._validator_ip_whitelist = _parse_validator_ip_whitelist(whitelist_raw)
        #     if not self._validator_ip_whitelist:
        #         raise RuntimeError(
        #             "MINER_VALIDATOR_IP_WHITELIST is empty; set entries like 63.142.193.28:31711 or 'off' to disable"
        #         )
        #     logger.info(f"[MINER] Validator IP whitelist: {whitelist_raw}")

        self._save_challenge_images = SAVE_CHALLENGE_IMAGES
        self._challenge_save_dir = Path(CHALLENGE_SAVE_DIR or "data/challenges")
        if not self._challenge_save_dir.is_absolute():
            self._challenge_save_dir = Path(__file__).resolve().parents[1] / self._challenge_save_dir
        if self._save_challenge_images:
            self._challenge_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[MINER] Saving challenge images to {self._challenge_save_dir}")

        self._challenge_r2: ChallengeR2Uploader | None = None
        self._challenge_persist_executor: concurrent.futures.ThreadPoolExecutor | None = None
        if R2_UPLOAD_CHALLENGES:
            try:
                self._challenge_r2 = ChallengeR2Uploader.from_env()
            except Exception as exc:
                raise RuntimeError(f"Failed to configure R2 challenge upload: {exc}") from exc
            logger.info(
                f"[MINER] R2 challenge upload enabled bucket={self._challenge_r2.bucket} "
                f"prefix={self._challenge_r2.prefix}"
            )

        if self._save_challenge_images or self._challenge_r2 is not None:
            worker_count = max(1, int(os.getenv("PERTURB_CHALLENGE_PERSIST_WORKERS", "2")))
            self._challenge_persist_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="challenge-persist",
            )
            logger.info(
                "[MINER] Challenge image persist runs in background after attack "
                f"(workers={worker_count})"
            )

        axon_port = int(getattr(getattr(self.config, "axon", object()), "port", 9000))
        self.axon = _make_axon(wallet=self.wallet, port=axon_port)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    def _persist_clean_challenge_image(
        self,
        task_id: str,
        clean_b64: str,
        prompt: str,
        true_label: str,
    ) -> None:
        if not clean_b64.strip():
            logger.warning(f"[CHALLENGE_SAVE] task_id={task_id} skipped: missing clean_image_b64")
            return

        safe_id = sanitize_task_id_for_filename(task_id)
        filename = f"{safe_id}_clean.jpg"

        try:
            jpeg_bytes = challenge_image_b64_to_jpeg_bytes(clean_b64)
        except Exception as exc:
            logger.warning(f"[CHALLENGE_SAVE] task_id={task_id} decode failed: {exc}")
            return

        if self._save_challenge_images:
            output_path = self._challenge_save_dir / filename
            try:
                output_path.write_bytes(jpeg_bytes)
            except Exception as exc:
                logger.warning(f"[CHALLENGE_SAVE] task_id={task_id} local write failed: {exc}")
            else:
                logger.info(f"[CHALLENGE_SAVED] task_id={task_id} path={output_path}")

        if self._challenge_r2 is not None:
            try:
                object_key = self._challenge_r2.upload_jpeg(
                    filename,
                    jpeg_bytes,
                    metadata={
                        "task_id": task_id,
                        "prompt": prompt,
                        "true_label": true_label,
                    },
                )
            except Exception as exc:
                logger.warning(f"[CHALLENGE_R2] task_id={task_id} upload failed: {exc}")
            else:
                logger.info(
                    f"[CHALLENGE_R2] task_id={task_id} bucket={self._challenge_r2.bucket} key={object_key}"
                )

    def _schedule_challenge_persist_background(
        self,
        synapse: AttackChallenge,
        task_id: str,
    ) -> None:
        if self._challenge_persist_executor is None:
            return
        clean_b64 = getattr(synapse, "clean_image_b64", None)
        if not isinstance(clean_b64, str) or not clean_b64.strip():
            return
        prompt = getattr(synapse, "prompt", "") or ""
        true_label = getattr(synapse, "true_label", "") or ""
        self._challenge_persist_executor.submit(
            self._persist_clean_challenge_image,
            task_id,
            clean_b64,
            prompt,
            true_label,
        )
        logger.info(f"[CHALLENGE_PERSIST] task_id={task_id} queued background")

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        task_id = getattr(synapse, "task_id", "unknown")
        self._log_step_start(
            "miner_forward",
            task_id=task_id,
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )
        logger.info(
            "[CHALLENGE_META] %s",
            json.dumps(
                {
                    "task_id": task_id,
                    "prompt": getattr(synapse, "prompt", "") or "",
                    "true_label": getattr(synapse, "true_label", "") or "",
                },
                ensure_ascii=False,
            ),
        )
        try:
            if synapse.norm_type != "Linf":
                logger.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                return synapse

            clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
            target_index = resolve_target_index(synapse.true_label)
            if target_index is None:
                logger.warning(
                    f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unresolved true_label={getattr(synapse, 'true_label', None)}"
                )
                synapse.perturbed_image_b64 = synapse.clean_image_b64
                return synapse

            epsilon = float(synapse.epsilon)
            min_delta = float(getattr(synapse, "min_delta", 0.002))

            result = run_attack(
                clean=clean,
                model=self.model,
                target_index=target_index,
                epsilon=epsilon,
                min_delta=min_delta,
                device=self.device,
            )
            synapse.perturbed_image_b64 = encode_image_b64(result.adv)
            logger.info(
                f"Finished task={getattr(synapse, 'task_id', 'unknown')} "
                f"algorithm={result.algorithm_name} key={result.algorithm_key} "
                f"target_idx={target_index} final_pred={result.pred_index} "
                f"best_delta={result.linf:.6f} best_rmse={result.rmse:.6f} min_delta={min_delta:.6f}"
            )
            return synapse
        finally:
            self._schedule_challenge_persist_background(synapse, task_id)

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        caller_ip = getattr(getattr(synapse, "dendrite", None), "ip", None)
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
            caller_ip=caller_ip,
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        # caller_uid: int | None = None
        # if hotkey in self.metagraph.hotkeys:
        #     caller_uid = self.metagraph.hotkeys.index(hotkey)

        # Validator IP whitelist — disabled until real validator endpoint is confirmed.
        # if self._validator_ip_whitelist:
        #     allowed, reason = _caller_ip_whitelisted(
        #         caller_ip=str(caller_ip) if caller_ip else None,
        #         caller_uid=caller_uid,
        #         metagraph=self.metagraph,
        #         whitelist=self._validator_ip_whitelist,
        #     )
        #     if not allowed:
        #         logger.warning(f"Blacklist reject: {reason}")
        #         return True, reason

        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey} ip={caller_ip}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} "
            f"with netuid: {self.config.netuid} axon_port={getattr(self.config.axon, 'port', 'unknown')}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = _resolve_axon_port(config)

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

