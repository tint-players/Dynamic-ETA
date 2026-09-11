from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session import SimulationSession, sessions
from .viz import config_for_visualization


app = FastAPI(title="Dynamic-ETA Simulator Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    scenario_name: str = "delhi_agra_corridor.yaml"


class PlaybackCommand(BaseModel):
    command: Literal["play", "pause", "reset", "step", "set_speed"]
    speed: float | None = Field(default=None, gt=0)


ALLOWED_PLAYBACK_SPEEDS = {0.5, 1.0, 2.0, 5.0, 10.0}


def _session_or_404(session_id: str) -> SimulationSession:
    try:
        return sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _frame_message(frame) -> dict:
    return {"type": "telemetry", "frame": frame.model_dump(mode="json")}


def _export_message(session: SimulationSession) -> dict | None:
    if session.export_paths is None:
        return None
    return {"type": "export_complete", "paths": session.export_paths}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios() -> dict:
    return {"scenarios": sessions.list_scenarios()}


@app.post("/api/sessions", status_code=201)
def create_session(request: CreateSessionRequest) -> dict:
    try:
        session = sessions.create(request.scenario_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "session_id": session.session_id,
        "scenario_id": session.scenario_id,
        "scenario_name": session.scenario_name,
        "config": config_for_visualization(session.config),
        "initial_frame": session.snapshot().model_dump(mode="json"),
    }


@app.get("/api/sessions/{session_id}/config")
def get_session_config(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return {
        "session_id": session.session_id,
        "scenario_id": session.scenario_id,
        "scenario_name": session.scenario_name,
        "config": config_for_visualization(session.config),
    }


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    session = _session_or_404(session_id)
    frame = session.reset()
    return {"session_id": session.session_id, "frame": frame.model_dump(mode="json")}


async def _send_state(websocket: WebSocket, session: SimulationSession) -> None:
    await websocket.send_json(
        {
            "type": "playback_state",
            "playing": session.playing,
            "playback_speed": session.playback_speed,
            "complete": session.engine.is_complete,
        }
    )


async def _send_export_if_ready(websocket: WebSocket, session: SimulationSession) -> None:
    message = _export_message(session)
    if message is not None:
        await websocket.send_json(message)


async def _handle_command(websocket: WebSocket, session: SimulationSession, payload: dict) -> None:
    try:
        command = PlaybackCommand.model_validate(payload)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Invalid command: {exc}"})
        return

    if command.command == "play":
        if not session.engine.is_complete:
            session.playing = True
    elif command.command == "pause":
        session.playing = False
    elif command.command == "set_speed":
        if command.speed not in ALLOWED_PLAYBACK_SPEEDS:
            await websocket.send_json(
                {"type": "error", "message": "speed must be one of 0.5, 1, 2, 5, 10"}
            )
            return
        session.playback_speed = float(command.speed)
    elif command.command == "reset":
        frame = session.reset()
        await websocket.send_json(_frame_message(frame))
    elif command.command == "step":
        session.playing = False
        frames = session.tick()
        if frames:
            await websocket.send_json(_frame_message(frames[-1]))
        await _send_export_if_ready(websocket, session)

    await _send_state(websocket, session)


@app.websocket("/ws/{session_id}")
async def simulation_websocket(websocket: WebSocket, session_id: str) -> None:
    try:
        session = sessions.get(session_id)
    except KeyError:
        await websocket.close(code=4404, reason="Unknown session")
        return

    await websocket.accept()
    await websocket.send_json(_frame_message(session.snapshot()))
    await _send_state(websocket, session)
    await _send_export_if_ready(websocket, session)

    try:
        while True:
            if session.playing and not session.engine.is_complete:
                # Playback speed changes wall-clock pacing only. Simulation tick_seconds
                # remains untouched, preserving identical physics at every UI speed.
                delay = session.config.simulation.tick_seconds / session.playback_speed
                try:
                    payload = await asyncio.wait_for(websocket.receive_json(), timeout=delay)
                    await _handle_command(websocket, session, payload)
                except asyncio.TimeoutError:
                    frames = session.tick()
                    if frames:
                        await websocket.send_json(_frame_message(frames[-1]))
                    if session.engine.is_complete:
                        session.playing = False
                        await _send_export_if_ready(websocket, session)
                        await _send_state(websocket, session)
            else:
                payload = await websocket.receive_json()
                await _handle_command(websocket, session, payload)
    except WebSocketDisconnect:
        session.playing = False
