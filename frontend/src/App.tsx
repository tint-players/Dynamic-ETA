import { useEffect, useRef, useState } from 'react'

import { connectSimulation, createSession, sendCommand } from './api'
import Dashboard, { type DebugEvent } from './components/Dashboard'
import type { PlaybackState, SessionCreateResponse, SocketMessage, TelemetryFrame } from './types'

const initialPlayback: PlaybackState = {
  playing: false,
  playback_speed: 1,
  complete: false,
}

function transitionEvents(previous: TelemetryFrame | null, current: TelemetryFrame): DebugEvent[] {
  if (!previous) return [{ id: `start-${current.tick}`, time: current.sim_time_s, text: `Simulation ready in ${current.current_block_id}` }]
  const events: DebugEvent[] = []
  const add = (suffix: string, text: string) => {
    events.push({ id: `${current.tick}-${suffix}`, time: current.sim_time_s, text })
  }

  if (previous.current_block_id !== current.current_block_id) {
    add('block', `Entered ${current.current_block_id}`)
  }
  if (previous.control_action !== current.control_action || previous.control_reason !== current.control_reason) {
    add('control', `${current.control_action}: ${current.control_reason}`)
  }
  if (previous.speed_kmh > 0.2 && current.speed_kmh <= 0.2) {
    add('stop', `Train stopped (${current.control_reason})`)
  }
  if (previous.speed_kmh <= 0.2 && current.speed_kmh > 0.2) {
    add('resume', 'Train resumed')
  }
  if (current.distance_to_destination_m === 0 && previous.distance_to_destination_m > 0) {
    add('destination', 'Reached destination')
  }
  return events
}

export default function App() {
  const [session, setSession] = useState<SessionCreateResponse | null>(null)
  const [frame, setFrame] = useState<TelemetryFrame | null>(null)
  const [history, setHistory] = useState<TelemetryFrame[]>([])
  const [events, setEvents] = useState<DebugEvent[]>([])
  const [playback, setPlayback] = useState<PlaybackState>(initialPlayback)
  const [connection, setConnection] = useState('connecting')
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const previousFrameRef = useRef<TelemetryFrame | null>(null)

  useEffect(() => {
    let cancelled = false

    const start = async () => {
      try {
        const created = await createSession()
        if (cancelled) return
        setSession(created)
        setFrame(created.initial_frame)
        setHistory([created.initial_frame])
        previousFrameRef.current = created.initial_frame

        socketRef.current = connectSimulation(
          created.session_id,
          (message: SocketMessage) => {
            if (message.type === 'telemetry') {
              const next = message.frame
              const isReset = next.tick === 0 && next.sim_time_s === 0 && (previousFrameRef.current?.tick ?? 0) > 0
              if (isReset) {
                setHistory([next])
                setEvents([{ id: 'reset-0', time: 0, text: 'Simulation reset' }])
              } else {
                setHistory((items) => {
                  if (items.at(-1)?.tick === next.tick && items.at(-1)?.sim_time_s === next.sim_time_s) return items
                  return [...items, next]
                })
                const additions = transitionEvents(previousFrameRef.current, next)
                if (additions.length) setEvents((items) => [...items, ...additions].slice(-250))
              }
              previousFrameRef.current = next
              setFrame(next)
            } else if (message.type === 'playback_state') {
              setPlayback({
                playing: message.playing,
                playback_speed: message.playback_speed,
                complete: message.complete,
              })
            } else if (message.type === 'error') {
              setError(message.message)
            }
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
    return () => {
      cancelled = true
      socketRef.current?.close()
    }
  }, [])

  if (error && !session) {
    return (
      <div className="boot-screen error-screen">
        <h1>Simulator dashboard could not start</h1>
        <p>{error}</p>
        <p>Confirm the FastAPI backend is running on port 8000, then refresh this page.</p>
      </div>
    )
  }

  if (!session || !frame) {
    return <div className="boot-screen"><div className="spinner" /><p>Starting simulator session…</p></div>
  }

  return (
    <>
      {error && <div className="error-toast" onClick={() => setError(null)}>{error}</div>}
      <Dashboard
        config={session.config}
        frame={frame}
        history={history}
        events={events}
        playback={playback}
        connection={connection}
        onPlay={() => sendCommand(socketRef.current, 'play')}
        onPause={() => sendCommand(socketRef.current, 'pause')}
        onReset={() => sendCommand(socketRef.current, 'reset')}
        onStep={() => sendCommand(socketRef.current, 'step')}
        onSpeed={(speed) => sendCommand(socketRef.current, 'set_speed', speed)}
      />
    </>
  )
}
