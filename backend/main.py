from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from simulator.models import MaintenanceType, SignalAspect, WeatherCondition

from .live_eta import live_eta_state
from .session import SimulationSession, sessions
from .viz import config_for_visualization


app = FastAPI(title="Dynamic-ETA Simulator Dashboard API", version="0.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://dynamic-hnvbaojse-tint-players.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    scenario_name: str = "delhi_agra_corridor.yaml"


class PlaybackCommand(BaseModel):
    command: Literal["play", "pause", "reset", "reset_constraints", "step", "set_speed"]
    speed: float | None = Field(default=None, gt=0)


class WeatherInjectionCommand(BaseModel):
    command: Literal["inject_weather"]
    block_id: str
    condition: WeatherCondition
    visibility_m: float = Field(gt=0)
    duration_s: float | None = Field(default=300.0, gt=0)


class SignalInjectionCommand(BaseModel):
    command: Literal["inject_signal"]
    signal_id: str
    aspect: SignalAspect
    duration_s: float | None = Field(default=60.0, gt=0)


class SignalResetCommand(BaseModel):
    command: Literal["reset_signal"]
    signal_id: str


class SpeedRestrictionInjectionCommand(BaseModel):
    command: Literal["inject_tsr", "inject_maintenance"]
    block_id: str
    track_id: str
    start_position_m: float = Field(ge=0)
    end_position_m: float = Field(gt=0)
    speed_limit_kmh: float = Field(gt=0)
    maintenance_type: MaintenanceType = MaintenanceType.SPEED_RESTRICTION
    duration_s: float | None = Field(default=300.0, gt=0)


ALLOWED_PLAYBACK_SPEEDS = {0.5, 1.0, 2.0, 5.0, 10.0}


def _session_or_404(session_id: str) -> SimulationSession:
    try:
        return sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _telemetry_message(session: SimulationSession, frames) -> dict:
    materialized = list(frames)
    return {
        "type": "telemetry_batch",
        "frames": [frame.model_dump(mode="json") for frame in materialized],
        "signal_states": session.signal_states(),
        "crossing_states": session.crossing_states(),
        "eta_state": live_eta_state(session, materialized),
    }


def _constraint_state_message(session: SimulationSession) -> dict:
    return {"type": "constraint_state", "state": session.active_constraints()}


def _export_message(session: SimulationSession) -> dict | None:
    if session.export_paths is None:
        return None
    return {"type": "export_complete", "paths": session.export_paths}


def _config_message(session: SimulationSession) -> dict:
    return {"type": "config_update", "config": config_for_visualization(session.config)}


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
    initial_frames = session.snapshots()
    return {
        "session_id": session.session_id,
        "scenario_id": session.scenario_id,
        "scenario_name": session.scenario_name,
        "config": config_for_visualization(session.config),
        "initial_frame": initial_frames[0].model_dump(mode="json"),
        "initial_frames": [frame.model_dump(mode="json") for frame in initial_frames],
        "signal_states": session.signal_states(),
        "crossing_states": session.crossing_states(),
        "constraint_state": session.active_constraints(),
        "eta_state": live_eta_state(session, initial_frames),
    }


@app.get("/api/sessions/{session_id}/config")
def get_session_config(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return {
        "session_id": session.session_id,
        "scenario_id": session.scenario_id,
        "scenario_name": session.scenario_name,
        "config": config_for_visualization(session.config),
        "constraint_state": session.active_constraints(),
    }


async def _send_state(websocket: WebSocket, session: SimulationSession) -> None:
    await websocket.send_json({
        "type": "playback_state",
        "playing": session.playing,
        "playback_speed": session.playback_speed,
        "complete": session.engine.is_complete,
    })


async def _send_export_if_ready(websocket: WebSocket, session: SimulationSession) -> None:
    message = _export_message(session)
    if message is not None:
        await websocket.send_json(message)


async def _send_live_state(websocket: WebSocket, session: SimulationSession, frames) -> None:
    await websocket.send_json(_telemetry_message(session, frames))
    await websocket.send_json(_constraint_state_message(session))


async def _handle_injection(websocket: WebSocket, session: SimulationSession, payload: dict) -> bool:
    command_name = payload.get("command")
    if command_name == "inject_weather":
        try:
            command = WeatherInjectionCommand.model_validate(payload)
            message = session.inject_weather(command.block_id, command.condition, command.visibility_m, command.duration_s)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid weather injection: {exc}"})
            return True
    elif command_name == "inject_signal":
        try:
            command = SignalInjectionCommand.model_validate(payload)
            message = session.inject_signal(command.signal_id, command.aspect, command.duration_s)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid signal injection: {exc}"})
            return True
    elif command_name == "reset_signal":
        try:
            command = SignalResetCommand.model_validate(payload)
            message = session.reset_signal(command.signal_id)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid signal reset: {exc}"})
            return True
    elif command_name in {"inject_tsr", "inject_maintenance"}:
        try:
            command = SpeedRestrictionInjectionCommand.model_validate(payload)
            kind = "tsr" if command.command == "inject_tsr" else "maintenance"
            message = session.inject_speed_restriction(
                kind=kind,
                block_id=command.block_id,
                track_id=command.track_id,
                start_position_m=command.start_position_m,
                end_position_m=command.end_position_m,
                speed_limit_kmh=command.speed_limit_kmh,
                duration_s=command.duration_s,
                maintenance_type=command.maintenance_type,
            )
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Invalid restriction injection: {exc}"})
            return True
    else:
        return False

    await websocket.send_json({"type": "injection_applied", "message": message, "sim_time_s": session.engine.sim_time_s})
    await websocket.send_json(_config_message(session))
    await _send_live_state(websocket, session, session.snapshots())
    return True


async def _handle_command(websocket: WebSocket, session: SimulationSession, payload: dict) -> None:
    if await _handle_injection(websocket, session, payload):
        await _send_state(websocket, session)
        return

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
            await websocket.send_json({"type": "error", "message": "speed must be one of 0.5, 1, 2, 5, 10"})
            return
        session.playback_speed = float(command.speed)
    elif command.command == "reset":
        frames = session.reset()
        await websocket.send_json(_config_message(session))
        await _send_live_state(websocket, session, frames)
    elif command.command == "reset_constraints":
        session.reset_constraints()
        await websocket.send_json({"type": "constraints_reset", "sim_time_s": session.engine.sim_time_s})
        await websocket.send_json(_config_message(session))
        await _send_live_state(websocket, session, session.snapshots())
    elif command.command == "step":
        session.playing = False
        frames = session.tick()
        if frames:
            await _send_live_state(websocket, session, frames)
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
    await _send_live_state(websocket, session, session.snapshots())
    await _send_state(websocket, session)
    await _send_export_if_ready(websocket, session)

    try:
        while True:
            if session.playing and not session.engine.is_complete:
                delay = session.config.simulation.tick_seconds / session.playback_speed
                try:
                    payload = await asyncio.wait_for(websocket.receive_json(), timeout=delay)
                    await _handle_command(websocket, session, payload)
                except asyncio.TimeoutError:
                    frames = session.tick()
                    if frames:
                        await _send_live_state(websocket, session, frames)
                    if session.engine.is_complete:
                        session.playing = False
                        await _send_export_if_ready(websocket, session)
                        await _send_state(websocket, session)
            else:
                payload = await websocket.receive_json()
                await _handle_command(websocket, session, payload)
    except WebSocketDisconnect:
        session.playing = False
