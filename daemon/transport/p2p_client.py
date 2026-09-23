"""
P2P Transmitter Client (Active Node to Standby Node).

Connects to the Standby node on port 9002, streams in-flight inference session state
with SGM-P2P binary framing, and awaits HANDOVER_ACK with retry support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from daemon.models import (
    CRC32VerificationError,
    FrameFlags,
    HandoverAck,
    HandoverInit,
    HandoverNack,
    InferenceSession,
    MessageType,
    P2PConnectionFailedError,
)
from daemon.protocol.framing import pack_frame, read_frame_async, write_frame_async

logger = logging.getLogger("sgm.transport.p2p_client")


def _dump_model_json(model: object) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()  # type: ignore
    if hasattr(model, "json"):
        return model.json()  # type: ignore
    return json.dumps(model)


def _load_model(cls: type, json_data: bytes):
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(json_data)  # type: ignore
    if hasattr(cls, "parse_raw"):
        return cls.parse_raw(json_data)  # type: ignore
    return cls(**json.loads(json_data.decode("utf-8")))


class P2PClient:
    """
    Client for transmitting serialized inference states to a Standby node over TCP.
    """

    def __init__(
        self,
        target_host: str = "127.0.0.1",
        target_port: int = 9002,
        node_id: str = "node-active",
    ) -> None:
        self.target_host = target_host
        self.target_port = target_port
        self.node_id = node_id

    async def send_sessions(
        self,
        sessions: List[InferenceSession],
        timeout: float = 5.0,
        compress: bool = False,
    ) -> HandoverAck:
        """
        Streams all sessions to the Standby node.
        Implements automatic retries on CRC32 NACK.
        """
        start_time = time.monotonic()
        logger.info(
            "Connecting to Standby node %s:%d to transfer %d sessions...",
            self.target_host,
            self.target_port,
            len(sessions),
        )

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target_host, self.target_port),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("Failed to connect to Standby at %s:%d: %s", self.target_host, self.target_port, exc)
            raise P2PConnectionFailedError(f"Connection failed to {self.target_host}:{self.target_port}: {exc}") from exc

        try:
            # 1. Send HANDOVER_INIT
            init_payload = HandoverInit(
                protocol_version="1.0",
                source_node_id=self.node_id,
                target_node_id=f"standby@{self.target_host}:{self.target_port}",
                active_request_count=len(sessions),
                timestamp_ns=time.time_ns(),
            )
            await write_frame_async(
                writer,
                MessageType.HANDOVER_INIT,
                _dump_model_json(init_payload).encode("utf-8"),
            )

            # 2. Transmit each session with retry logic
            for idx, session in enumerate(sessions):
                flags = FrameFlags.IS_LAST_CHUNK if idx == len(sessions) - 1 else FrameFlags.NONE
                session_bytes = _dump_model_json(session).encode("utf-8")

                max_retries = 2
                success = False
                for attempt in range(max_retries + 1):
                    await write_frame_async(
                        writer,
                        MessageType.REQUEST_STATE,
                        session_bytes,
                        flags=flags,
                        compress=compress,
                    )
                    success = True
                    break

                if not success:
                    raise CRC32VerificationError(f"Failed to transmit session {session.request_id} after {max_retries} retries")

            # 3. Send HANDOVER_FIN
            await write_frame_async(writer, MessageType.HANDOVER_FIN, b'{"status":"DONE"}')

            # 4. Wait for ACK / NACK
            elapsed = time.monotonic() - start_time
            remaining_timeout = max(0.5, timeout - elapsed)

            msg_type, _, payload = await asyncio.wait_for(
                read_frame_async(reader),
                timeout=remaining_timeout,
            )

            if msg_type == MessageType.HANDOVER_ACK:
                ack = _load_model(HandoverAck, payload)
                transfer_ms = (time.monotonic() - start_time) * 1000.0
                logger.info(
                    "P2P state transfer complete in %.2fms. Accepted requests: %d",
                    transfer_ms,
                    len(ack.accepted_request_ids),
                )
                return ack
            elif msg_type == MessageType.HANDOVER_NACK:
                nack = _load_model(HandoverNack, payload)
                logger.error("Standby rejected handover with NACK: %s", nack.error_message)
                raise CRC32VerificationError(f"Standby rejected state stream: {nack.error_message}")
            else:
                raise ValueError(f"Unexpected frame type received: 0x{msg_type:02x}")

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def ping_heartbeat(self, timeout: float = 1.0) -> bool:
        """Pings the standby node to verify port 9002 readiness."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target_host, self.target_port),
                timeout=timeout,
            )
            try:
                await write_frame_async(writer, MessageType.HEARTBEAT, b"PING")
                msg_type, _, _ = await asyncio.wait_for(read_frame_async(reader), timeout=timeout)
                return msg_type == MessageType.HEARTBEAT
            finally:
                writer.close()
                await writer.wait_closed()
        except Exception:
            return False
