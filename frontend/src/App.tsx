import { useEffect, useRef, useState } from 'react'

import { connectSimulation, createSession, sendCommand, sendInjection } from './api'
import Dashboard, { type DebugEvent } from './components/Dashboard'
import type { CrossingState, ExportPaths, PlaybackState, SessionCreateResponse, SignalAspect, SocketMessage, TelemetryFrame } from './types'

const initialPlayback: PlaybackState = { playing: false, playback_speed: 1, complete: false }

function trainEvents(previous: TelemetryFrame | undefined, current: TelemetryFrame): DebugEvent[] {
  if (!previous) return [{ id: `${current.train_id}-start`, time: current.sim_time_s, text: `${current.train_id} ready on ${current.track_id}` }]
  const events: DebugEvent[] = []
  const add = (suffix: string, text: string) => events.push({ id: `${current.train_id}-${current.tick}-${suffix}`, time: current.sim_time_s, text })
  if (previous.current_block_id !== current.current_block_id) add('block', `${current.train_id} entered ${current.current_block_id}`)
  if (previous.track_id !== current.track_id) add('track', `${current.train_id}: ${previous.track_id} → ${current.track_id}`)
  if (previous.control_action !== current.control_action || previous.control_reason !== current.control_reason) add('control', `${current.train_id}: ${current.control_action} · ${current.control_reason}`)
  if (previous.current_station_id !== current.current_station_id && current.current_station_id) add('station', `${current.train_id} dwelling at ${current.current_station_id}`)
  if (!previous.completed && current.completed) add('destination', `${current.train_id} reached destination`)
  return events
}

export default function App() {
  const [session, setSession] = useState<SessionCreateResponse | null>(null)
  const [frames, setFrames] = useState<TelemetryFrame[]>([])
  const [historyByTrain, setHistoryByTrain] = useState<Record<string, TelemetryFrame[]>>({})
  const [selectedTrainId, setSelectedTrainId] = useState<string>('')
  const [events, setEvents] = useState<DebugEvent[]>([])
  const [signalStates, setSignalStates] = useState<Record<string, SignalAspect>>({})
  const [crossingStates, setCrossingStates] = useState<Record<string, CrossingState>>({})
  const [playback, setPlayback] = useState<PlaybackState>(initialPlayback)
  const [exportPaths, setExportPaths] = useState<ExportPaths | null>(null)
  const [connection, setConnection] = useState('connecting')
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const previousFramesRef = useRef<Record<string, TelemetryFrame>>({})

  useEffect(() => {
    let cancelled = false
    const start = async () => {
      try {
        const created = await createSession()
        if (cancelled) return
        setSession(created)
        setFrames(created.initial_frames)
        setSignalStates(created.signal_states)
        setCrossingStates(created.crossing_states)
        const initialSelected = created.config.train.train_id
        setSelectedTrainId(initialSelected)
        setHistoryByTrain(Object.fromEntries(created.initial_frames.map((f) => [f.train_id, [f]])))
        setEvents(created.initial_frames.map((f) => ({ id: `${f.train_id}-start`, time: 0, text: `${f.train_id} ready on ${f.track_id}` })))
        previousFramesRef.current = Object.fromEntries(created.initial_frames.map((f) => [f.train_id, f]))

        socketRef.current = connectSimulation(
          created.session_id,
          (message: SocketMessage) => {
            if (message.type === 'telemetry_batch') {
              const nextFrames = message.frames
              const isReset = nextFrames.every((f) => f.tick === 0 && f.sim_time_s === 0) && Object.values(previousFramesRef.current).some((f) => f.tick > 0)
              setFrames(nextFrames)
              setSignalStates(message.signal_states)
              setCrossingStates(message.crossing_states)
              if (isReset) {
                setHistoryByTrain(Object.fromEntries(nextFrames.map((f) => [f.train_id, [f]])))
                setEvents([{ id: 'reset-0', time: 0, text: 'Simulation reset to clean baseline' }])
                setExportPaths(null)
              } else {
                const additions = nextFrames.flatMap((next) => trainEvents(previousFramesRef.current[next.train_id], next))
                if (additions.length) setEvents((items) => [...items, ...additions].slice(-400))
                setHistoryByTrain((current) => {
                  const updated = { ...current }
                  for (const next of nextFrames) {
                    const history = updated[next.train_id] ?? []
                    updated[next.train_id] = history.at(-1)?.tick === next.tick ? history : [...history, next]
                  }
                  return updated
                })
              }
              previousFramesRef.current = Object.fromEntries(nextFrames.map((f) => [f.train_id, f]))
            } else if (message.type === 'playback_state') {
              setPlayback({ playing: message.playing, playback_speed: message.playback_speed, complete: message.complete })
            } else if (message.type === 'config_update') {
              setSession((current) => current ? { ...current, config: message.config } : current)
            } else if (message.type === 'injection_applied') {
              setEvents((items) => [
                ...items,
                { id: `inject-${message.sim_time_s}-${items.length}`, time: message.sim_time_s, text: message.message },
              ].slice(-400))
            } else if (message.type === 'export_complete') {
              setExportPaths(message.paths)
              const time = Object.values(previousFramesRef.current)[0]?.sim_time_s ?? 0
              setEvents((items) => [...items, { id: `export-${message.paths.csv}`, time, text: `Dataset exported: ${message.paths.csv}` }].slice(-400))
            } else if (message.type === 'error') setError(message.message)
          },
          setConnection,
        )
      } catch (cause) {
        if (!cancelled) {
          setConnection('error')
          setError(cause instanceof Error ? cause.message : 'Unable to start simulator session')
        }
      }
    }
    void start()
    return () => { cancelled = true; socketRef.current?.close() }
  }, [])

  if (error && !session) return <div className="boot-screen error-screen"><h1>Simulator dashboard could not start</h1><p>{error}</p><p>Confirm the FastAPI backend is running on port 8000, then refresh this page.</p></div>
  if (!session || frames.length === 0) return <div className="boot-screen"><div className="spinner" /><p>Starting multi-train simulator…</p></div>

  const selectedFrame = frames.find((f) => f.train_id === selectedTrainId) ?? frames[0]
  const selectedHistory = historyByTrain[selectedFrame.train_id] ?? [selectedFrame]
  return (
    <>
      {error && <div className="error-toast" onClick={() => setError(null)}>{error}</div>}
      <Dashboard
        config={session.config}
        frame={selectedFrame}
        frames={frames}
        selectedTrainId={selectedFrame.train_id}
        onSelectTrain={setSelectedTrainId}
        signalStates={signalStates}
        crossingStates={crossingStates}
        history={selectedHistory}
        events={events}
        playback={playback}
        exportPaths={exportPaths}
        connection={connection}
        onPlay={() => sendCommand(socketRef.current, 'play')}
        onPause={() => sendCommand(socketRef.current, 'pause')}
        onReset={() => sendCommand(socketRef.current, 'reset')}
        onStep={() => sendCommand(socketRef.current, 'step')}
        onSpeed={(speed) => sendCommand(socketRef.current, 'set_speed', speed)}
        onInject={(request) => sendInjection(socketRef.current, request)}
      />
    </>
  )
}
