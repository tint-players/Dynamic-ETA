import { useState } from 'react'

import type { CrossingState, ExportPaths, ManualInjectionRequest, PlaybackState, SignalAspect, SimulatorConfigViz, TelemetryFrame, WeatherCondition } from '../types'
import RailwayRouteV2 from './RailwayRouteV2'

export interface DebugEvent { id: string; time: number; text: string }

interface DashboardProps {
  config: SimulatorConfigViz
  frame: TelemetryFrame
  frames: TelemetryFrame[]
  selectedTrainId: string
  onSelectTrain: (trainId: string) => void
  signalStates: Record<string, SignalAspect>
  crossingStates: Record<string, CrossingState>
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
  onInject: (request: ManualInjectionRequest) => void
}

const fmt = (value: number | null | undefined, digits = 1) => value == null ? '—' : value.toFixed(digits)

function SimulationControls({ playback, connection, onPlay, onPause, onReset, onStep, onSpeed }: DashboardProps) {
  const speeds = [0.5, 1, 2, 5, 10]
  return <section className="controls panel"><div className="transport-buttons"><button onClick={playback.playing ? onPause : onPlay} className="primary">{playback.playing ? 'Pause' : 'Play'}</button><button onClick={onStep} disabled={playback.playing || playback.complete}>Step</button><button onClick={onReset}>Reset baseline</button></div><div className="speed-buttons"><span>Playback</span>{speeds.map((speed) => <button key={speed} className={playback.playback_speed === speed ? 'selected' : ''} onClick={() => onSpeed(speed)}>{speed}×</button>)}</div><div className={`connection ${connection}`}>● {connection}</div></section>
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
          <button type="button" className={`fleet-card ${train.active ? 'active' : ''} ${train.completed ? 'completed' : ''} ${train.train_id === selectedTrainId ? 'selected-train' : ''}`} key={train.train_id} onClick={() => onSelectTrain(train.train_id)}>
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

function WeatherInjector({ config, disabled, onInject }: Pick<DashboardProps, 'config' | 'onInject'> & { disabled: boolean }) {
  const [blockId, setBlockId] = useState(config.route.blocks[0]?.block_id ?? '')
  const [condition, setCondition] = useState<WeatherCondition>('RAIN')
  const [visibility, setVisibility] = useState(5000)

  return (
    <div className="inject-card">
      <div className="inject-card-heading"><strong>Weather</strong><span>applies immediately</span></div>
      <label>Block<select value={blockId} onChange={(event) => setBlockId(event.target.value)}>{config.route.blocks.map((block) => <option key={block.block_id} value={block.block_id}>{block.block_id}</option>)}</select></label>
      <label>Condition<select value={condition} onChange={(event) => setCondition(event.target.value as WeatherCondition)}><option value="CLEAR">CLEAR</option><option value="RAIN">RAIN</option><option value="HEAVY_RAIN">HEAVY RAIN</option><option value="FOG">FOG</option><option value="HEAVY_FOG">HEAVY FOG</option></select></label>
      <label className="slider-label"><span>Visibility <b>{visibility} m</b></span><input type="range" min={100} max={20000} step={100} value={visibility} onChange={(event) => setVisibility(Number(event.target.value))} /></label>
      <button disabled={disabled || !blockId || visibility <= 0} onClick={() => onInject({ command: 'inject_weather', block_id: blockId, condition, visibility_m: visibility })}>Apply weather now</button>
    </div>
  )
}

function RestrictionInjector({ config, disabled, onInject, kind }: Pick<DashboardProps, 'config' | 'onInject'> & { disabled: boolean; kind: 'tsr' | 'maintenance' }) {
  const firstBlock = config.route.blocks[0]
  const [blockId, setBlockId] = useState(firstBlock?.block_id ?? '')
  const [startM, setStartM] = useState(0)
  const [endM, setEndM] = useState(firstBlock?.length_m ?? 1)
  const [limit, setLimit] = useState(kind === 'tsr' ? 70 : 45)
  const [duration, setDuration] = useState(300)
  const block = config.route.blocks.find((item) => item.block_id === blockId)

  const selectBlock = (next: string) => {
    setBlockId(next)
    const selected = config.route.blocks.find((item) => item.block_id === next)
    setStartM(0)
    if (selected) setEndM(selected.length_m)
  }

  const valid = Boolean(block) && startM >= 0 && endM > startM && endM <= (block?.length_m ?? 0) && limit > 0 && duration > 0
  const title = kind === 'tsr' ? 'Temporary speed restriction' : 'Maintenance speed limit'

  return (
    <div className="inject-card">
      <div className="inject-card-heading"><strong>{title}</strong><span>speed limit only</span></div>
      <label>Block<select value={blockId} onChange={(event) => selectBlock(event.target.value)}>{config.route.blocks.map((item) => <option key={item.block_id} value={item.block_id}>{item.block_id} · {item.length_m}m</option>)}</select></label>
      <div className="inject-pair"><label>Start (m)<input type="number" min={0} max={block?.length_m} value={startM} onChange={(event) => setStartM(Number(event.target.value))} /></label><label>End (m)<input type="number" min={1} max={block?.length_m} value={endM} onChange={(event) => setEndM(Number(event.target.value))} /></label></div>
      <label>Limit (km/h)<input type="number" min={1} value={limit} onChange={(event) => setLimit(Number(event.target.value))} /></label>
      <label className="slider-label"><span>Duration <b>{duration} s</b></span><input type="range" min={10} max={1800} step={10} value={duration} onChange={(event) => setDuration(Number(event.target.value))} /></label>
      <button disabled={disabled || !valid} onClick={() => onInject({ command: kind === 'tsr' ? 'inject_tsr' : 'inject_maintenance', block_id: blockId, start_position_m: startM, end_position_m: endM, speed_limit_kmh: limit, duration_s: duration })}>Apply {kind === 'tsr' ? 'TSR' : 'maintenance limit'} now</button>
    </div>
  )
}

function SignalInjector({ config, signalStates, disabled, onInject }: Pick<DashboardProps, 'config' | 'signalStates' | 'onInject'> & { disabled: boolean }) {
  const [signalId, setSignalId] = useState(config.signals[0]?.signal_id ?? '')
  const [aspect, setAspect] = useState<SignalAspect>('RED')
  const [duration, setDuration] = useState(60)
  const signal = config.signals.find((item) => item.signal_id === signalId)

  return (
    <div className="inject-card">
      <div className="inject-card-heading"><strong>Signal override</strong><span>temporary manual aspect</span></div>
      <label>Signal<select value={signalId} onChange={(event) => setSignalId(event.target.value)}>{config.signals.map((item) => <option key={item.signal_id} value={item.signal_id}>{item.signal_id} · {item.protected_block_id} · {item.track_id}</option>)}</select></label>
      <label>Force aspect<select value={aspect} onChange={(event) => setAspect(event.target.value as SignalAspect)}><option value="GREEN">GREEN</option><option value="YELLOW">YELLOW</option><option value="RED">RED</option></select></label>
      <div className="inject-current">Live now: <strong>{signalStates[signalId] ?? '—'}</strong>{signal && <span> protects {signal.protected_block_id}</span>}</div>
      <label className="slider-label"><span>Duration <b>{duration} s</b></span><input type="range" min={10} max={600} step={10} value={duration} onChange={(event) => setDuration(Number(event.target.value))} /></label>
      <button disabled={disabled || !signalId} onClick={() => onInject({ command: 'inject_signal', signal_id: signalId, aspect, duration_s: duration })}>Force signal now</button>
      <small className="inject-warning">Manual aspect temporarily overrides occupancy-driven signalling for this signal only.</small>
    </div>
  )
}

function LiveBlockConstraints({ config, signalStates, crossingStates, simTime }: Pick<DashboardProps, 'config' | 'signalStates' | 'crossingStates'> & { simTime: number }) {
  const activeWeather = (blockId: string) => {
    const schedule = config.environment.weather.find((item) => item.block_id === blockId)
    if (!schedule) return { condition: 'CLEAR', visibility_m: 10000 }
    return [...schedule.timeline].reverse().find((item) => item.start_time_s <= simTime) ?? schedule.timeline[0]
  }
  const activeRestrictions = (blockId: string, kind: 'tsr' | 'maintenance') => {
    const source = kind === 'tsr' ? config.environment.temporary_speed_restrictions : config.environment.maintenance_restrictions
    return source.filter((item) => item.block_id === blockId && item.start_time_s <= simTime && (item.end_time_s == null || simTime < item.end_time_s))
  }

  return (
    <div className="live-constraint-section">
      <div className="inject-card-heading"><strong>Live constraints by block</strong><span>simulation time {fmt(simTime, 0)} s</span></div>
      <div className="live-block-grid">
        {config.route.blocks.map((block) => {
          const weather = activeWeather(block.block_id)
          const tsrs = activeRestrictions(block.block_id, 'tsr')
          const maintenance = activeRestrictions(block.block_id, 'maintenance')
          const signals = config.signals.filter((item) => item.protected_block_id === block.block_id)
          const crossings = config.environment.crossings.filter((item) => item.block_id === block.block_id)
          return <div className="live-block-card" key={block.block_id}>
            <div className="live-block-title"><strong>{block.block_id}</strong><span>MAX {block.speed_limit_kmh} km/h</span></div>
            <div><span>Weather</span><b>{weather.condition.replaceAll('_', ' ')} · {weather.visibility_m}m</b></div>
            <div><span>TSR</span><b>{tsrs.length ? tsrs.map((item) => `${item.speed_limit_kmh} km/h @ ${item.start_position_m}-${item.end_position_m}m`).join(' · ') : 'none'}</b></div>
            <div><span>Maintenance</span><b>{maintenance.length ? maintenance.map((item) => `${item.speed_limit_kmh} km/h @ ${item.start_position_m}-${item.end_position_m}m`).join(' · ') : 'none'}</b></div>
            <div><span>Signals</span><b>{signals.length ? signals.map((item) => `${item.signal_id}:${signalStates[item.signal_id] ?? '—'}`).join(' · ') : 'none'}</b></div>
            <div><span>Crossing</span><b>{crossings.length ? crossings.map((item) => `${item.crossing_id}:${crossingStates[item.crossing_id] ?? '—'}`).join(' · ') : 'none'}</b></div>
          </div>
        })}
      </div>
    </div>
  )
}

function ManualInjectionPanel({ config, playback, signalStates, crossingStates, frame, onInject }: Pick<DashboardProps, 'config' | 'playback' | 'signalStates' | 'crossingStates' | 'frame' | 'onInject'>) {
  return (
    <section className="panel injection-panel">
      <div className="panel-heading"><div><span className="eyebrow small-eyebrow">Phase 2 · manual variables</span><h2>Constraint console</h2></div><span>{config.environment.temporary_speed_restrictions.length} TSR · {config.environment.maintenance_restrictions.length} maintenance</span></div>
      <div className="injection-note">Changes apply at the current simulation second and are captured in tickwise CSV/Parquet. Reset baseline removes manual weather, TSR, maintenance and signal overrides.</div>
      <LiveBlockConstraints config={config} signalStates={signalStates} crossingStates={crossingStates} simTime={frame.sim_time_s} />
      <div className="inject-grid"><WeatherInjector config={config} disabled={playback.complete} onInject={onInject} /><RestrictionInjector config={config} disabled={playback.complete} onInject={onInject} kind="tsr" /><RestrictionInjector config={config} disabled={playback.complete} onInject={onInject} kind="maintenance" /><SignalInjector config={config} signalStates={signalStates} disabled={playback.complete} onInject={onInject} /></div>
    </section>
  )
}

function TrainStatePanel({ frame, config }: { frame: TelemetryFrame; config: SimulatorConfigViz }) {
  const selectedTrainConfig = config.trains.find((item) => item.train.train_id === frame.train_id)?.train ?? (config.train.train_id === frame.train_id ? config.train : undefined)
  const trainMaxSpeed = selectedTrainConfig?.max_speed_kmh
  return <section className="panel"><div className="panel-heading"><h2>{frame.train_id} state</h2><span className="action-badge">{frame.control_action}</span></div><div className="metric-grid"><Metric label="Simulation time" value={`${fmt(frame.sim_time_s, 0)} s`} /><Metric label="Track / direction" value={`${frame.track_id} · ${frame.direction}`} /><Metric label="Current block" value={frame.current_block_id} sub={`${fmt(frame.position_in_block_m, 0)} m in block`} /><Metric label="Speed" value={`${fmt(frame.speed_kmh)} km/h`} /><Metric label="Train max speed" value={trainMaxSpeed == null ? '—' : `${fmt(trainMaxSpeed, 0)} km/h`} /><Metric label="Acceleration" value={`${fmt(frame.acceleration_ms2, 2)} m/s²`} /><Metric label="Route position" value={`${fmt(frame.route_position_m, 0)} m`} /><Metric label="Distance to destination" value={`${fmt(frame.distance_to_destination_m, 0)} m`} /><Metric label="Progress" value={`${(frame.route_progress * 100).toFixed(1)}%`} /><Metric label="Control reason" value={frame.control_reason} /><Metric label="Station" value={frame.current_station_id ?? '—'} /><Metric label="Effective ceiling" value={`${fmt(frame.effective_speed_ceiling_kmh)} km/h`} /><Metric label="Weather" value={frame.weather.replaceAll('_', ' ')} sub={`${fmt(frame.visibility_m, 0)} m visibility`} /></div><div className="ground-truth-note">Ground Truth Remaining Time: {frame.actual_remaining_time_s == null ? 'unavailable during live run' : `${fmt(frame.actual_remaining_time_s, 0)} s`}</div></section>
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
  const [constraintsOpen, setConstraintsOpen] = useState(false)
  const { config, frame, frames, selectedTrainId, onSelectTrain, signalStates, crossingStates, history, events, playback, exportPaths } = props
  return <main className="dashboard"><header className="topbar"><div><span className="eyebrow">Component A · multi-train network simulation</span><h1>Dynamic-ETA Railway Simulator</h1></div><div className="topbar-actions"><button className={`constraints-toggle ${constraintsOpen ? 'open' : ''}`} onClick={() => setConstraintsOpen((current) => !current)}>Constraints {constraintsOpen ? '▲' : '▼'}</button><div className="scenario-card"><span>{frame.scenario_id}</span><strong>{frames.length} live trains · {config.stations.length} stations</strong><small>{config.dynamic_signalling ? 'occupancy-driven signalling' : 'scheduled signalling'}</small></div></div></header><SimulationControls {...props} />{constraintsOpen && <ManualInjectionPanel config={config} playback={playback} signalStates={signalStates} crossingStates={crossingStates} frame={frame} onInject={props.onInject} />}<RailwayRouteV2 config={config} frames={frames} signalStates={signalStates} crossingStates={crossingStates} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} /><FleetPanel frames={frames} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} /><div className="two-column"><TrainStatePanel frame={frame} config={config} /><ConstraintPanel frame={frame} /></div><SpeedChart history={history} /><EventLog events={events} />{playback.complete && <div className="complete-banner">All trains complete at {frame.sim_time_s.toFixed(0)} s{exportPaths && <div>Dataset exported · CSV: {exportPaths.csv} · Parquet: {exportPaths.parquet}</div>}</div>}</main>
}
