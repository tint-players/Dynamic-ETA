import type { PlaybackState, SignalAspect, SimulatorConfigViz, TelemetryFrame } from '../types'

export interface DebugEvent {
  id: string
  time: number
  text: string
}

interface DashboardProps {
  config: SimulatorConfigViz
  frame: TelemetryFrame
  history: TelemetryFrame[]
  events: DebugEvent[]
  playback: PlaybackState
  connection: string
  onPlay: () => void
  onPause: () => void
  onReset: () => void
  onStep: () => void
  onSpeed: (speed: number) => void
}

const fmt = (value: number | null | undefined, digits = 1) =>
  value == null ? '—' : value.toFixed(digits)

function activeAt<T extends { start_time_s: number }>(timeline: T[], simTime: number): T | undefined {
  let active: T | undefined
  for (const entry of timeline) {
    if (entry.start_time_s <= simTime) active = entry
    else break
  }
  return active
}

function signalAspect(config: SimulatorConfigViz, signalId: string, simTime: number): SignalAspect {
  const schedule = config.environment.signal_states.find((item) => item.signal_id === signalId)
  return activeAt(schedule?.timeline ?? [], simTime)?.aspect ?? 'GREEN'
}

function RailwayRoute({ config, frame }: { config: SimulatorConfigViz; frame: TelemetryFrame }) {
  const total = config.route.total_length_m
  const x = (metres: number) => `${(metres / total) * 100}%`

  return (
    <section className="panel route-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Live infrastructure view</span>
          <h2>{config.route.route_name}</h2>
        </div>
        <div className="route-meta">{(total / 1000).toFixed(1)} km total</div>
      </div>
      <div className="route-scroll">
        <div className="route-canvas">
          <div className="track-line" />

          {config.route.blocks.map((block) => (
            <div
              className="block-segment"
              key={block.block_id}
              style={{ left: x(block.route_start_m), width: x(block.length_m) }}
            >
              <div className="block-label">
                <strong>{block.block_id}</strong>
                <span>{block.speed_limit_kmh} km/h</span>
                {block.curve_speed_limit_kmh != null && <span>curve {block.curve_speed_limit_kmh} km/h</span>}
              </div>
            </div>
          ))}

          {config.environment.temporary_speed_restrictions.map((item) => (
            <div
              className="restriction tsr"
              key={item.restriction_id}
              style={{ left: x(item.route_start_m), width: x(item.route_end_m - item.route_start_m) }}
              title={`${item.restriction_id}: ${item.speed_limit_kmh} km/h`}
            >
              TSR {item.speed_limit_kmh}
            </div>
          ))}

          {config.environment.maintenance_restrictions.map((item) => (
            <div
              className="restriction maintenance"
              key={item.restriction_id}
              style={{ left: x(item.route_start_m), width: x(item.route_end_m - item.route_start_m) }}
              title={`${item.restriction_id}: ${item.speed_limit_kmh} km/h`}
            >
              MNT {item.speed_limit_kmh}
            </div>
          ))}

          {config.signals.map((signal) => {
            const aspect = signalAspect(config, signal.signal_id, frame.sim_time_s)
            return (
              <div
                className="signal-marker"
                key={signal.signal_id}
                style={{ left: x(signal.route_position_m) }}
                title={`${signal.signal_id} — ${aspect}`}
              >
                <span className={`signal-light ${aspect.toLowerCase()}`} />
                <span>{signal.signal_id}</span>
              </div>
            )
          })}

          {config.environment.crossings.map((crossing) => {
            const state = activeAt(crossing.timeline, frame.sim_time_s)?.state ?? 'OPEN_FOR_TRAIN'
            return (
              <div
                className={`crossing-marker ${state === 'CLOSED_FOR_TRAIN' ? 'closed' : ''}`}
                key={crossing.crossing_id}
                style={{ left: x(crossing.route_position_m) }}
                title={`${crossing.crossing_id}: ${state}`}
              >
                ×
                <span>{crossing.crossing_id}</span>
              </div>
            )
          })}

          <div className="endpoint source" style={{ left: x(config.journey.source_route_m) }}>SOURCE</div>
          <div className="endpoint destination" style={{ left: x(config.journey.destination_route_m) }}>DEST</div>

          <div className="train-marker" style={{ left: x(frame.route_position_m) }} title={`${frame.speed_kmh.toFixed(1)} km/h`}>
            <div className="train-icon">▰</div>
            <div className="train-caption">{frame.speed_kmh.toFixed(1)} km/h</div>
          </div>
        </div>
      </div>
      <div className="legend">
        <span><i className="legend-swatch tsr-swatch" />TSR</span>
        <span><i className="legend-swatch maintenance-swatch" />Maintenance</span>
        <span>● signals</span>
        <span>× crossing</span>
      </div>
    </section>
  )
}

function SimulationControls({ playback, connection, onPlay, onPause, onReset, onStep, onSpeed }: Omit<DashboardProps, 'config' | 'frame' | 'history' | 'events'>) {
  const speeds = [0.5, 1, 2, 5, 10]
  return (
    <section className="controls panel">
      <div className="transport-buttons">
        <button onClick={playback.playing ? onPause : onPlay} className="primary">
          {playback.playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={onStep} disabled={playback.playing || playback.complete}>Step</button>
        <button onClick={onReset}>Reset</button>
      </div>
      <div className="speed-buttons">
        <span>Playback</span>
        {speeds.map((speed) => (
          <button
            key={speed}
            className={playback.playback_speed === speed ? 'selected' : ''}
            onClick={() => onSpeed(speed)}
          >
            {speed}×
          </button>
        ))}
      </div>
      <div className={`connection ${connection}`}>● {connection}</div>
    </section>
  )
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  )
}

function TrainStatePanel({ frame }: { frame: TelemetryFrame }) {
  return (
    <section className="panel">
      <div className="panel-heading"><h2>Train state</h2><span className="action-badge">{frame.control_action}</span></div>
      <div className="metric-grid">
        <Metric label="Simulation time" value={`${fmt(frame.sim_time_s, 0)} s`} />
        <Metric label="Current block" value={frame.current_block_id} sub={`${fmt(frame.position_in_block_m, 0)} m in block`} />
        <Metric label="Speed" value={`${fmt(frame.speed_kmh)} km/h`} />
        <Metric label="Acceleration" value={`${fmt(frame.acceleration_ms2, 2)} m/s²`} />
        <Metric label="Route position" value={`${fmt(frame.route_position_m, 0)} m`} />
        <Metric label="Distance to destination" value={`${fmt(frame.distance_to_destination_m, 0)} m`} />
        <Metric label="Progress" value={`${(frame.route_progress * 100).toFixed(1)}%`} />
        <Metric label="Control reason" value={frame.control_reason} />
        <Metric label="Block limit" value={`${fmt(frame.current_block_speed_limit_kmh)} km/h`} />
        <Metric label="Effective ceiling" value={`${fmt(frame.effective_speed_ceiling_kmh)} km/h`} />
        <Metric label="Curve limit" value={frame.current_curve_speed_limit_kmh == null ? '—' : `${fmt(frame.current_curve_speed_limit_kmh)} km/h`} />
        <Metric label="Weather" value={frame.weather.replaceAll('_', ' ')} sub={`${fmt(frame.visibility_m, 0)} m visibility`} />
      </div>
      <div className="ground-truth-note">
        Ground Truth Remaining Time: {frame.actual_remaining_time_s == null ? 'unavailable during live run' : `${fmt(frame.actual_remaining_time_s, 0)} s`}
      </div>
    </section>
  )
}

function ConstraintPanel({ frame }: { frame: TelemetryFrame }) {
  const rows = [
    ['Next signal', frame.next_signal_aspect ?? '—', frame.distance_to_next_signal_m],
    ['Second signal', frame.second_signal_aspect ?? '—', frame.distance_to_second_signal_m],
    ['Next speed change', frame.next_speed_limit_kmh == null ? '—' : `${fmt(frame.next_speed_limit_kmh)} km/h`, frame.distance_to_next_speed_change_m],
    ['Next curve', frame.next_curve_limit_kmh == null ? '—' : `${fmt(frame.next_curve_limit_kmh)} km/h`, frame.distance_to_next_curve_m],
    ['Next TSR', frame.next_tsr_limit_kmh == null ? '—' : `${fmt(frame.next_tsr_limit_kmh)} km/h`, frame.distance_to_next_tsr_m],
    ['Next crossing', frame.next_crossing_state?.replaceAll('_', ' ') ?? '—', frame.distance_to_next_crossing_m],
  ] as const

  return (
    <section className="panel">
      <div className="panel-heading"><h2>Controller lookahead</h2></div>
      <div className="constraint-list">
        {rows.map(([name, value, distance]) => (
          <div className="constraint-row" key={name}>
            <span>{name}</span>
            <strong>{value}</strong>
            <small>{distance == null ? '—' : `${fmt(distance, 0)} m ahead`}</small>
          </div>
        ))}
      </div>
    </section>
  )
}

function SpeedChart({ history }: { history: TelemetryFrame[] }) {
  const data = history.slice(-240)
  const width = 900
  const height = 230
  const pad = 30
  const maxY = Math.max(140, ...data.map((f) => Math.max(f.speed_kmh, f.effective_speed_ceiling_kmh)))
  const firstT = data[0]?.sim_time_s ?? 0
  const lastT = data[data.length - 1]?.sim_time_s ?? firstT + 1
  const spanT = Math.max(1, lastT - firstT)
  const point = (f: TelemetryFrame, value: number) => {
    const x = pad + ((f.sim_time_s - firstT) / spanT) * (width - pad * 2)
    const y = height - pad - (value / maxY) * (height - pad * 2)
    return `${x},${y}`
  }
  const speedPoints = data.map((f) => point(f, f.speed_kmh)).join(' ')
  const ceilingPoints = data.map((f) => point(f, f.effective_speed_ceiling_kmh)).join(' ')

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <h2>Speed vs effective ceiling</h2>
        <span className="chart-key"><i className="actual-line" />Actual <i className="ceiling-line" />Ceiling</span>
      </div>
      <svg className="speed-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Speed chart">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis" />
        <text x={4} y={pad + 4}>{maxY.toFixed(0)}</text>
        <text x={4} y={height - pad + 4}>0</text>
        <text x={pad} y={height - 7}>{firstT.toFixed(0)}s</text>
        <text x={width - 65} y={height - 7}>{lastT.toFixed(0)}s</text>
        <polyline points={ceilingPoints} className="ceiling-polyline" />
        <polyline points={speedPoints} className="speed-polyline" />
      </svg>
    </section>
  )
}

function EventLog({ events }: { events: DebugEvent[] }) {
  return (
    <section className="panel event-panel">
      <div className="panel-heading"><h2>Event log</h2><span>{events.length} events</span></div>
      <div className="event-list">
        {events.length === 0 && <div className="empty">Events will appear as backend state changes.</div>}
        {[...events].reverse().map((event) => (
          <div className="event-row" key={event.id}>
            <time>{event.time.toFixed(0)}s</time>
            <span>{event.text}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export default function Dashboard(props: DashboardProps) {
  const { config, frame, history, events, playback } = props
  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <span className="eyebrow">Component A · simulation observability</span>
          <h1>Dynamic-ETA Railway Simulator</h1>
        </div>
        <div className="scenario-card">
          <span>{frame.scenario_id}</span>
          <strong>{config.train.train_name || config.train.train_id}</strong>
          <small>seed {config.simulation.random_seed ?? '—'}</small>
        </div>
      </header>

      <SimulationControls {...props} />
      <RailwayRoute config={config} frame={frame} />

      <div className="two-column">
        <TrainStatePanel frame={frame} />
        <ConstraintPanel frame={frame} />
      </div>

      <SpeedChart history={history} />
      <EventLog events={events} />

      {playback.complete && <div className="complete-banner">Journey complete — destination reached at {frame.sim_time_s.toFixed(0)} s</div>}
    </main>
  )
}
