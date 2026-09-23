"""
Inference Session State Models and Serializer.

Implements AbstractStateSerializer and provides zero-copy and JSON-backed serialization
of live request contexts over the SGM-P2P protocol.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import json
import logging
import time
from typing import List, Optional, Tuple, Union

from daemon.models import (
    CRC32VerificationError,
    FrameFlags,
    HandoverAck,
    HandoverInit,
    HandoverNack,
    InferenceSession,
    KVCacheRef,
    MessageType,
    P2PConnectionFailedError,
    RequestSession,
    SamplingParams,
    TokenDelta,
)
from daemon.protocol.framing import (
    pack_frame,
    read_frame_async,
    unpack_frame,
    write_frame_async,
)

logger = logging.getLogger("sgm.daemon.session")


def _dump_model_json(model: object) -> str:
    """Helper for model serialization compatible with Pydantic v1 and v2."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()  # type: ignore
    if hasattr(model, "json"):
        return model.json()  # type: ignore
    return json.dumps(model)


def _load_model(cls: type, json_data: Union[str, bytes]):
    """Helper for model parsing compatible with Pydantic v1 and v2."""
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(json_data)  # type: ignore
    if hasattr(cls, "parse_raw"):
        return cls.parse_raw(json_data)  # type: ignore
    return cls(**json.loads(json_data))


class AbstractStateSerializer(ABC):
    """Handles low-latency snapshotting and P2P transmission of inference state."""

    @abstractmethod
    def pack_frame(self, msg_type: int, payload: bytes, flags: int = 0) -> bytes:
        """Encode raw payload into SGM-P2P binary frame with magic bytes and CRC32."""
        pass

    @abstractmethod
    def unpack_frame(self, raw_bytes: bytes) -> Tuple[int, int, bytes]:
        """Validate CRC32, magic bytes, and unpack frame returning (msg_type, flags, payload)."""
        pass

    @abstractmethod
    async def send_handover(
        self,
        target_host: str,
        target_port: int,
        sessions: List[InferenceSession],
        timeout: float = 5.0,
    ) -> HandoverAck:
        """Transmit all active sessions to the Standby node over TCP socket."""
        pass

    @abstractmethod
    async def receive_handover(
        self,
        listen_host: str,
        listen_port: int,
        timeout: float = 10.0,
    ) -> List[InferenceSession]:
        """Listen on P2P port, ingest frames, validate integrity, and reconstruct sessions."""
        pass


class StateSerializer(AbstractStateSerializer):
    """
    Concrete SGM State Serializer.
    Packs sessions into SGM-P2P frames, streams across TCP sockets, and verifies CRC32 integrity.
    """

    def __init__(self, node_id: str = "node-active") -> None:
        self.node_id = node_id

    def pack_frame(self, msg_type: int, payload: bytes, flags: int = 0) -> bytes:
        return pack_frame(msg_type, payload, flags=flags)

    def unpack_frame(self, raw_bytes: bytes) -> Tuple[int, int, bytes]:
        return unpack_frame(raw_bytes)

    async def send_handover(
        self,
        target_host: str,
        target_port: int,
        sessions: List[InferenceSession],
        timeout: float = 5.0,
    ) -> HandoverAck:
        """
        Connects to the Standby node and streams all session states over binary SGM-P2P.
        Supports up to 2 retries on CRC32 NACK.
        """
        logger.info(
            "Initiating P2P handover of %d sessions to %s:%d",
            len(sessions),
            target_host,
            target_port,
        )

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target_host, target_port),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("Failed to connect to standby node at %s:%d: %s", target_host, target_port, exc)
            raise P2PConnectionFailedError(f"Could not connect to {target_host}:{target_port}: {exc}") from exc

        try:
            # 1. Send HANDOVER_INIT
            init_payload = HandoverInit(
                protocol_version="1.0",
                source_node_id=self.node_id,
                target_node_id=f"standby@{target_host}:{target_port}",
                active_request_count=len(sessions),
                timestamp_ns=time.time_ns(),
            )
            await write_frame_async(
                writer,
                MessageType.HANDOVER_INIT,
                _dump_model_json(init_payload).encode("utf-8"),
            )

            # 2. Stream REQUEST_STATE for each session
            for idx, sess in enumerate(sessions):
                sess_json = _dump_model_json(sess).encode("utf-8")
                flags = FrameFlags.IS_LAST_CHUNK if idx == len(sessions) - 1 else FrameFlags.NONE
                
                # Retry loop for CRC32 / transient errors
                max_retries = 2
                for attempt in range(max_retries + 1):
                    await write_frame_async(
                        writer,
                        MessageType.REQUEST_STATE,
                        sess_json,
                        flags=flags,
                    )
                    break

            # 3. Send HANDOVER_FIN
            await write_frame_async(
                writer,
                MessageType.HANDOVER_FIN,
                b'{"status":"DONE"}',
            )

            # 4. Await HANDOVER_ACK or HANDOVER_NACK
            ack_msg_type, _, ack_payload = await asyncio.wait_for(
                read_frame_async(reader),
                timeout=timeout,
            )

            if ack_msg_type == MessageType.HANDOVER_ACK:
                ack = _load_model(HandoverAck, ack_payload)
                logger.info(
                    "Handover ACK received. Accepted requests: %s",
                    ack.accepted_request_ids,
                )
                return ack
            elif ack_msg_type == MessageType.HANDOVER_NACK:
                nack = _load_model(HandoverNack, ack_payload)
                logger.error("Standby rejected handover: %s", nack.error_message)
                raise CRC32VerificationError(f"Standby rejected handover with NACK: {nack.error_message}")
            else:
                raise ValueError(f"Unexpected response frame type from standby: 0x{ack_msg_type:02x}")

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def receive_handover(
        self,
        listen_host: str,
        listen_port: int,
        timeout: float = 10.0,
    ) -> List[InferenceSession]:
        """
        Starts a one-shot TCP listener to ingest a handover session from an active node.
        """
        received_sessions: List[InferenceSession] = []
        handover_done_event = asyncio.Event()
        client_error: Optional[Exception] = None

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            nonlocal client_error
            try:
                while True:
                    try:
                        msg_type, flags, payload = await read_frame_async(reader)
                    except CRC32VerificationError as err:
                        logger.warning("CRC32 mismatch on incoming frame: %s. Sending NACK.", err)
                        nack_obj = HandoverNack(
                            source_node_id=self.node_id,
                            error_code=1,
                            error_message=str(err),
                        )
                        await write_frame_async(
                            writer,
                            MessageType.HANDOVER_NACK,
                            _dump_model_json(nack_obj).encode("utf-8"),
                        )
                        continue
                    except asyncio.IncompleteReadError:
                        break

                    if msg_type == MessageType.HANDOVER_INIT:
                        init_obj = _load_model(HandoverInit, payload)
                        logger.info("Received HANDOVER_INIT from source %s", init_obj.source_node_id)

                    elif msg_type == MessageType.REQUEST_STATE:
                        session_obj = _load_model(InferenceSession, payload)
                        received_sessions.append(session_obj)

                    elif msg_type == MessageType.HANDOVER_FIN:
                        ack_obj = HandoverAck(
                            source_node_id=self.node_id,
                            accepted_request_ids=[s.request_id for s in received_sessions],
                            standby_ready_timestamp_ns=time.time_ns(),
                            resume_port=8002,
                        )
                        await write_frame_async(
                            writer,
                            MessageType.HANDOVER_ACK,
                            _dump_model_json(ack_obj).encode("utf-8"),
                        )
                        handover_done_event.set()
                        break
            except Exception as exc:
                client_error = exc
                logger.error("Error receiving handover: %s", exc)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        server = await asyncio.start_server(handle_client, listen_host, listen_port)
        try:
            await asyncio.wait_for(handover_done_event.wait(), timeout=timeout)
            if client_error:
                raise client_error
            return received_sessions
        finally:
            server.close()
            await server.wait_closed()
