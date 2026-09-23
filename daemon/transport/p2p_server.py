"""
P2P Receiver Server (Standby Node Port 9002).

Accepts binary SGM-P2P connections from dying Active nodes, verifies CRC32 checksums,
deserializes in-flight inference session state, sends HANDOVER_ACK, and delivers sessions
to the standby engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional

from daemon.models import (
    CRC32VerificationError,
    HandoverAck,
    HandoverInit,
    HandoverNack,
    InferenceSession,
    MessageType,
)
from daemon.protocol.framing import read_frame_async, write_frame_async

logger = logging.getLogger("sgm.transport.p2p_server")

SessionHandlerCallback = Callable[[List[InferenceSession]], Awaitable[None]]


def _load_model(cls: type, json_data: bytes):
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(json_data)  # type: ignore
    if hasattr(cls, "parse_raw"):
        return cls.parse_raw(json_data)  # type: ignore
    return cls(**json.loads(json_data.decode("utf-8")))


def _dump_model_json(model: object) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()  # type: ignore
    if hasattr(model, "json"):
        return model.json()  # type: ignore
    return json.dumps(model)


class P2PServer:
    """
    TCP server for high-speed P2P state streaming on the Standby node.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9002,
        node_id: str = "node-standby",
        resume_port: int = 8002,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self.resume_port = resume_port
        self._server: Optional[asyncio.Server] = None
        self._session_callbacks: List[SessionHandlerCallback] = []
        self._received_sessions: Dict[str, InferenceSession] = {}
        self._running: bool = False

    def register_session_callback(self, callback: SessionHandlerCallback) -> None:
        """Registers a callback invoked when a batch of sessions finishes handover."""
        self._session_callbacks.append(callback)

    async def start(self) -> None:
        """Starts the P2P TCP server."""
        if self._running:
            return
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self._running = True
        logger.info("P2P Receiver Server listening on %s:%d (node_id=%s)", self.host, self.port, self.node_id)

    async def stop(self) -> None:
        """Stops the P2P TCP server and closes all active sockets."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("P2P Receiver Server stopped.")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("P2P connection accepted from %s", peer)
        sessions_in_flight: List[InferenceSession] = []
        source_node_id = "unknown"

        try:
            while self._running:
                try:
                    msg_type, flags, payload = await read_frame_async(reader)
                except CRC32VerificationError as crc_err:
                    logger.error("CRC32 verification failed on frame: %s. Sending HANDOVER_NACK.", crc_err)
                    nack = HandoverNack(
                        source_node_id=self.node_id,
                        error_code=0x01,
                        error_message=f"CRC32 checksum mismatch: {crc_err}",
                    )
                    await write_frame_async(
                        writer,
                        MessageType.HANDOVER_NACK,
                        _dump_model_json(nack).encode("utf-8"),
                    )
                    continue
                except asyncio.IncompleteReadError:
                    logger.debug("P2P client %s closed connection", peer)
                    break

                if msg_type == MessageType.HANDOVER_INIT:
                    init_data = _load_model(HandoverInit, payload)
                    source_node_id = init_data.source_node_id
                    logger.info(
                        "Received HANDOVER_INIT from %s (expecting %d requests)",
                        source_node_id,
                        init_data.active_request_count,
                    )

                elif msg_type == MessageType.REQUEST_STATE:
                    session = _load_model(InferenceSession, payload)
                    sessions_in_flight.append(session)
                    self._received_sessions[session.request_id] = session
                    logger.debug(
                        "Ingested state for request %s (tokens: %d, last_flushed: %d)",
                        session.request_id,
                        session.total_tokens_generated,
                        session.last_flushed_sequence_id,
                    )

                elif msg_type == MessageType.HANDOVER_FIN:
                    logger.info(
                        "Received HANDOVER_FIN. Successfully ingested %d sessions from %s.",
                        len(sessions_in_flight),
                        source_node_id,
                    )
                    
                    # Notify standby callbacks so standby engine can warm KV / schedule requests
                    if self._session_callbacks and sessions_in_flight:
                        await asyncio.gather(
                            *(cb(sessions_in_flight) for cb in self._session_callbacks),
                            return_exceptions=True,
                        )

                    ack = HandoverAck(
                        source_node_id=self.node_id,
                        accepted_request_ids=[s.request_id for s in sessions_in_flight],
                        standby_ready_timestamp_ns=time.time_ns(),
                        resume_port=self.resume_port,
                    )
                    await write_frame_async(
                        writer,
                        MessageType.HANDOVER_ACK,
                        _dump_model_json(ack).encode("utf-8"),
                    )
                    break

                elif msg_type == MessageType.HEARTBEAT:
                    # Echo heartbeat back
                    await write_frame_async(writer, MessageType.HEARTBEAT, b"PONG")

        except Exception as exc:
            logger.error("Exception handling P2P client %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
