import type { ExportPaths, PlaybackState, SignalAspect, SimulatorConfigViz, TelemetryFrame } from '../types'
import RailwayRouteV2 from './RailwayRouteV2'

export interface DebugEvent { id: string; time: number; text: string }

interface DashboardProps {
  config: SimulatorConfigViz
  frame: TelemetryFrame
  frames: TelemetryFrame[]
  selectedTrainId: string
  onSelectTrain: (trainId: string) => void
  signalStates: Record<string, SignalAspect>
  history: TelemetryFrame[]
  events: DebugEvent[]
  playback: PlaybackState
  exportPaths: ExportPaths | null
  connection: string
  onPlay: () => void
  onPause: () => void
  onReset: () => void
  onStep: () => void
  onSpeed: (speed: number) => void
}

const fmt = (value: number | null | undefined, digits = 1) => value == null ? '—' : value.toFixed(digits)

function SimulationControls({ playback, connection, onPlay, onPause, onReset, onStep, onSpeed }: DashboardProps) {
  const speeds = [0.5, 1, 2, 5, 10]
  return <section className="controls panel"><div className="transport-buttons"><button onClick={playback.playing ? onPause : onPlay} className="primary">{playback.playing ? 'Pause' : 'Play'}</button><button onClick={onStep} disabled={playback.playing || playback.complete}>Step</button><button onClick={onReset}>Reset</button></div><div className="speed-buttons"><span>Playback</span>{speeds.map((speed) => <button key={speed} className={playback.playback_speed === speed ? 'selected' : ''} onClick={() => onSpeed(speed)}>{speed}×</button>)}</div><div className={`connection ${connection}`}>● {connection}</div></section>
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</div>
}

function FleetPanel({ frames, selectedTrainId, onSelectTrain }: Pick<DashboardProps, 'frames' | 'selectedTrainId' | 'onSelectTrain'>) {
  return (
    <section className="panel fleet-panel">
      <div className="panel-heading"><h2>Live fleet</h2><span>Click a train for details</span></div>
      <div className="fleet-grid">
        {frames.map((train) => (
          <button
            type="button"
            className={`fleet-card ${train.active ? 'active' : ''} ${train.completed ? 'completed' : ''} ${train.train_id === selectedTrainId ? 'selected-train' : ''}`}
            key={train.train_id}
            onClick={() => onSelectTrain(train.train_id)}
          >
            <div><strong>{train.train_id}</strong><small>{train.track_id} · {train.direction}</small></div>
            <b>{train.speed_kmh.toFixed(1)} km/h</b>
            <span>{train.current_block_id} · {(train.route_progress * 100).toFixed(0)}%</span>
            <span>{train.current_station_id ? `At ${train.current_station_id}` : train.control_reason}</span>
          </button>
        ))}
      </div>
    </section>
  )
}

function TrainStatePanel({ frame }: { frame: TelemetryFrame }) {
  return <section className="panel"><div className="panel-heading"><h2>{frame.train_id} state</h2><span className="action-badge">{frame.control_action}</span></div><div className="metric-grid"><Metric label="Simulation time" value={`${fmt(frame.sim_time_s, 0)} s`} /><Metric label="Track / direction" value={`${frame.track_id} · ${frame.direction}`} /><Metric label="Current block" value={frame.current_block_id} sub={`${fmt(frame.position_in_block_m, 0)} m in block`} /><Metric label="Speed" value={`${fmt(frame.speed_kmh)} km/h`} /><Metric label="Acceleration" value={`${fmt(frame.acceleration_ms2, 2)} m/s²`} /><Metric label="Route position" value={`${fmt(frame.route_position_m, 0)} m`} /><Metric label="Distance to destination" value={`${fmt(frame.distance_to_destination_m, 0)} m`} /><Metric label="Progress" value={`${(frame.route_progress * 100).toFixed(1)}%`} /><Metric label="Control reason" value={frame.control_reason} /><Metric label="Station" value={frame.current_station_id ?? '—'} /><Metric label="Effective ceiling" value={`${fmt(frame.effective_speed_ceiling_kmh)} km/h`} /><Metric label="Weather" value={frame.weather.replaceAll('_', ' ')} sub={`${fmt(frame.visibility_m, 0)} m visibility`} /></div><div className="ground-truth-note">Ground Truth Remaining Time: {frame.actual_remaining_time_s == null ? 'unavailable during live run' : `${fmt(frame.actual_remaining_time_s, 0)} s`}</div></section>
}

function ConstraintPanel({ frame }: { frame: TelemetryFrame }) {
  const rows = [['Next signal', frame.next_signal_aspect ?? '—', frame.distance_to_next_signal_m], ['Second signal', frame.second_signal_aspect ?? '—', frame.distance_to_second_signal_m], ['Next speed change', frame.next_speed_limit_kmh == null ? '—' : `${fmt(frame.next_speed_limit_kmh)} km/h`, frame.distance_to_next_speed_change_m], ['Next curve', frame.next_curve_limit_kmh == null ? '—' : `${fmt(frame.next_curve_limit_kmh)} km/h`, frame.distance_to_next_curve_m], ['Next TSR', frame.next_tsr_limit_kmh == null ? '—' : `${fmt(frame.next_tsr_limit_kmh)} km/h`, frame.distance_to_next_tsr_m], ['Next crossing', frame.next_crossing_state?.replaceAll('_', ' ') ?? '—', frame.distance_to_next_crossing_m]] as const
  return <section className="panel"><div className="panel-heading"><h2>{frame.train_id} lookahead</h2></div><div className="constraint-list">{rows.map(([name, value, distance]) => <div className="constraint-row" key={name}><span>{name}</span><strong>{value}</strong><small>{distance == null ? '—' : `${fmt(distance, 0)} m ahead`}</small></div>)}</div></section>
}

function SpeedChart({ history }: { history: TelemetryFrame[] }) {
  const data = history.slice(-240); const width = 900, height = 230, pad = 30
  const maxY = Math.max(140, ...data.map((f) => Math.max(f.speed_kmh, f.effective_speed_ceiling_kmh)))
  const firstT = data[0]?.sim_time_s ?? 0; const lastT = data[data.length - 1]?.sim_time_s ?? firstT + 1; const spanT = Math.max(1, lastT - firstT)
  const point = (f: TelemetryFrame, value: number) => `${pad + ((f.sim_time_s - firstT) / spanT) * (width - pad * 2)},${height - pad - (value / maxY) * (height - pad * 2)}`
  return <section className="panel chart-panel"><div className="panel-heading"><h2>{data.at(-1)?.train_id ?? 'Selected train'} speed vs effective ceiling</h2><span className="chart-key"><i className="actual-line" />Actual <i className="ceiling-line" />Ceiling</span></div><svg className="speed-chart" viewBox={`0 0 ${width} ${height}`}><line x1={pad} y1={height-pad} x2={width-pad} y2={height-pad} className="axis"/><line x1={pad} y1={pad} x2={pad} y2={height-pad} className="axis"/><polyline points={data.map((f) => point(f, f.effective_speed_ceiling_kmh)).join(' ')} className="ceiling-polyline"/><polyline points={data.map((f) => point(f, f.speed_kmh)).join(' ')} className="speed-polyline"/></svg></section>
}

function EventLog({ events }: { events: DebugEvent[] }) {
  return <section className="panel event-panel"><div className="panel-heading"><h2>Event log</h2><span>{events.length} events</span></div><div className="event-list">{[...events].reverse().map((event) => <div className="event-row" key={event.id}><time>{event.time.toFixed(0)}s</time><span>{event.text}</span></div>)}</div></section>
}

export default function Dashboard(props: DashboardProps) {
  const { config, frame, frames, selectedTrainId, onSelectTrain, signalStates, history, events, playback, exportPaths } = props
  return <main className="dashboard"><header className="topbar"><div><span className="eyebrow">Component A · multi-train network simulation</span><h1>Dynamic-ETA Railway Simulator</h1></div><div className="scenario-card"><span>{frame.scenario_id}</span><strong>{frames.length} live trains · {config.stations.length} stations</strong><small>{config.dynamic_signalling ? 'occupancy-driven signalling' : 'scheduled signalling'}</small></div></header><SimulationControls {...props} /><RailwayRouteV2 config={config} frames={frames} signalStates={signalStates} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} /><FleetPanel frames={frames} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} /><div className="two-column"><TrainStatePanel frame={frame} /><ConstraintPanel frame={frame} /></div><SpeedChart history={history} /><EventLog events={events} />{playback.complete && <div className="complete-banner">All trains complete at {frame.sim_time_s.toFixed(0)} s{exportPaths && <div>Dataset exported · CSV: {exportPaths.csv} · Parquet: {exportPaths.parquet}</div>}</div>}</main>
}
