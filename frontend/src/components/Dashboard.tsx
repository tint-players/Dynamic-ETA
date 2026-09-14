import { useMemo, useState } from 'react'

import type { ConstraintState, CrossingState, ExportPaths, MaintenanceType, ManualInjectionRequest, PlaybackState, SignalAspect, SimulatorConfigViz, TelemetryFrame, WeatherCondition } from '../types'
import RailwayRouteV2 from './RailwayRouteV2'

export interface DebugEvent { id: string; time: number; text: string }
export interface RangeDraft { kind: 'tsr' | 'maintenance'; block_id: string; track_id: string; start_position_m: number; end_position_m: number }
type ConstraintTool = 'weather' | 'tsr' | 'maintenance' | 'signal'

interface DashboardProps {
  config: SimulatorConfigViz
  frame: TelemetryFrame
  frames: TelemetryFrame[]
  selectedTrainId: string
  onSelectTrain: (trainId: string) => void
  signalStates: Record<string, SignalAspect>
  crossingStates: Record<string, CrossingState>
  constraintState: ConstraintState
  history: TelemetryFrame[]
  events: DebugEvent[]
  playback: PlaybackState
  exportPaths: ExportPaths | null
  connection: string
  onPlay: () => void
  onPause: () => void
  onReset: () => void
  onResetConstraints: () => void
  onStep: () => void
  onSpeed: (speed: number) => void
  onInject: (request: ManualInjectionRequest) => void
}

const fmt = (value: number | null | undefined, digits = 1) => value == null ? '—' : value.toFixed(digits)
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))
const WEATHER_VISIBILITY: Record<WeatherCondition, number> = { CLEAR: 10000, RAIN: 5000, HEAVY_RAIN: 2500, FOG: 1000, HEAVY_FOG: 500 }

function SimulationControls({ playback, connection, onPlay, onPause, onReset, onResetConstraints, onStep, onSpeed }: DashboardProps) {
  const speeds = [0.5, 1, 2, 5, 10]
  return <section className="controls panel"><div className="transport-buttons"><button onClick={playback.playing ? onPause : onPlay} className="primary">{playback.playing ? 'Pause' : 'Play'}</button><button onClick={onStep} disabled={playback.playing || playback.complete}>Step</button><button onClick={onReset}>Reset simulation</button><button onClick={onResetConstraints}>Reset constraints</button></div><div className="speed-buttons"><span>Playback</span>{speeds.map((speed) => <button key={speed} className={playback.playback_speed === speed ? 'selected' : ''} onClick={() => onSpeed(speed)}>{speed}×</button>)}</div><div className={`connection ${connection}`}>● {connection}</div></section>
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</div>
}

function FleetPanel({ frames, selectedTrainId, onSelectTrain }: Pick<DashboardProps, 'frames' | 'selectedTrainId' | 'onSelectTrain'>) {
  return <section className="panel fleet-panel"><div className="panel-heading"><h2>Live fleet</h2><span>Click a train for details</span></div><div className="fleet-grid">{frames.map((train) => <button type="button" className={`fleet-card ${train.active ? 'active' : ''} ${train.completed ? 'completed' : ''} ${train.train_id === selectedTrainId ? 'selected-train' : ''}`} key={train.train_id} onClick={() => onSelectTrain(train.train_id)}><div><strong>{train.train_id}</strong><small>{train.track_id} · {train.direction}</small></div><b>{train.speed_kmh.toFixed(1)} km/h</b><span>{train.current_block_id} · {(train.route_progress * 100).toFixed(0)}%</span><span>{train.current_station_id ? `At ${train.current_station_id}` : train.control_reason}</span></button>)}</div></section>
}

function Stepper({ label, value, min, max, step, suffix, onChange }: { label: string; value: number; min: number; max: number; step: number; suffix: string; onChange: (value: number) => void }) {
  const set = (next: number) => onChange(clamp(Math.round(next / step) * step, min, max))
  return <label className="compact-stepper"><span>{label}</span><div><button type="button" onClick={() => set(value - step)}>−</button><strong>{value}{suffix}</strong><button type="button" onClick={() => set(value + step)}>+</button></div></label>
}

function ConstraintDrawer({ config, simTime, playback, activeTool, onTool, selectedSignalId, onSelectedSignal, restrictionTrack, onRestrictionTrack, rangeDraft, onRangeDraft, onInject }: {
  config: SimulatorConfigViz
  simTime: number
  playback: PlaybackState
  activeTool: ConstraintTool
  onTool: (tool: ConstraintTool) => void
  selectedSignalId: string
  onSelectedSignal: (signalId: string) => void
  restrictionTrack: string
  onRestrictionTrack: (trackId: string) => void
  rangeDraft: RangeDraft | null
  onRangeDraft: (draft: RangeDraft | null) => void
  onInject: DashboardProps['onInject']
}) {
  const [weatherBlock, setWeatherBlock] = useState(config.route.blocks[0]?.block_id ?? '')
  const [weatherCondition, setWeatherCondition] = useState<WeatherCondition>('CLEAR')
  const [visibility, setVisibility] = useState(WEATHER_VISIBILITY.CLEAR)
  const [weatherDuration, setWeatherDuration] = useState(300)
  const [restrictionLimit, setRestrictionLimit] = useState(60)
  const [restrictionDuration, setRestrictionDuration] = useState(300)
  const [maintenanceType, setMaintenanceType] = useState<MaintenanceType>('SPEED_RESTRICTION')
  const [signalAspect, setSignalAspect] = useState<SignalAspect>('RED')
  const [signalDuration, setSignalDuration] = useState(120)
  const disabled = playback.complete

  const chooseWeather = (condition: WeatherCondition) => {
    setWeatherCondition(condition)
    setVisibility(WEATHER_VISIBILITY[condition])
  }

  const applyRestriction = () => {
    if (!rangeDraft || !restrictionTrack || rangeDraft.track_id !== restrictionTrack) return
    if (rangeDraft.kind === 'tsr') {
      onInject({ command: 'inject_tsr', block_id: rangeDraft.block_id, track_id: restrictionTrack, start_position_m: rangeDraft.start_position_m, end_position_m: rangeDraft.end_position_m, speed_limit_kmh: restrictionLimit, duration_s: restrictionDuration })
    } else {
      onInject({ command: 'inject_maintenance', block_id: rangeDraft.block_id, track_id: restrictionTrack, start_position_m: rangeDraft.start_position_m, end_position_m: rangeDraft.end_position_m, speed_limit_kmh: restrictionLimit, maintenance_type: maintenanceType, duration_s: restrictionDuration })
    }
    onRangeDraft(null)
  }

  return <aside className="constraints-drawer panel">
    <div className="drawer-heading"><div><span className="eyebrow small-eyebrow">CONSTRAINT CONTROL</span><h2>Manual intervention</h2></div><span>SIM {Math.floor(simTime / 60).toString().padStart(2, '0')}:{Math.floor(simTime % 60).toString().padStart(2, '0')}</span></div>
    <div className="drawer-tabs">{(['weather', 'tsr', 'maintenance', 'signal'] as ConstraintTool[]).map((tool) => <button key={tool} type="button" className={activeTool === tool ? 'active' : ''} onClick={() => onTool(tool)}>{tool === 'tsr' ? 'TSR' : tool === 'maintenance' ? 'Maintenance' : tool[0].toUpperCase() + tool.slice(1)}</button>)}</div>

    {activeTool === 'weather' && <div className="drawer-section"><div className="drawer-section-title"><strong>Weather</strong><span>timed by simulation clock</span></div><label className="drawer-field">Block<select value={weatherBlock} onChange={(event) => setWeatherBlock(event.target.value)}>{config.route.blocks.map((block) => <option key={block.block_id} value={block.block_id}>{block.block_id}</option>)}</select></label><label className="drawer-field">Condition<select value={weatherCondition} onChange={(event) => chooseWeather(event.target.value as WeatherCondition)}><option value="CLEAR">CLEAR</option><option value="RAIN">RAIN</option><option value="HEAVY_RAIN">HEAVY RAIN</option><option value="FOG">FOG</option><option value="HEAVY_FOG">HEAVY FOG</option></select></label><Stepper label="Visibility" value={visibility} min={100} max={20000} step={100} suffix=" m" onChange={setVisibility} /><Stepper label="Duration" value={weatherDuration} min={10} max={3600} step={10} suffix=" s" onChange={setWeatherDuration} /><button className="drawer-apply" disabled={disabled || !weatherBlock} onClick={() => onInject({ command: 'inject_weather', block_id: weatherBlock, condition: weatherCondition, visibility_m: visibility, duration_s: weatherDuration })}>Apply weather</button></div>}

    {(activeTool === 'tsr' || activeTool === 'maintenance') && <div className="drawer-section"><div className="drawer-section-title"><strong>{activeTool === 'tsr' ? 'Temporary speed restriction' : 'Maintenance'}</strong><span>block range + physical track</span></div><label className="drawer-field">Track<select value={restrictionTrack} onChange={(event) => onRestrictionTrack(event.target.value)}>{config.route.track_ids.map((trackId) => <option key={trackId} value={trackId}>{trackId}</option>)}</select></label>{activeTool === 'maintenance' && <label className="drawer-field">Mode<select value={maintenanceType} onChange={(event) => setMaintenanceType(event.target.value as MaintenanceType)}><option value="SPEED_RESTRICTION">SPEED RESTRICTION</option><option value="FULL_CLOSURE">FULL CLOSURE</option></select></label>}<div className={`range-readout ${rangeDraft?.kind === activeTool ? 'ready' : ''}`}>{rangeDraft?.kind === activeTool ? <><strong>{rangeDraft.track_id} · {rangeDraft.block_id}</strong><span>{Math.round(rangeDraft.start_position_m)}m → {Math.round(rangeDraft.end_position_m)}m</span><small>{Math.round(rangeDraft.end_position_m - rangeDraft.start_position_m)}m selected · drag either handle on the selected track</small></> : <><strong>No range selected</strong><span>Select a track, then click that railway track</span><small>A centered 200m section will be created only on the selected physical track.</small></>}</div>{rangeDraft?.kind === activeTool && <button type="button" className="drawer-remove" onClick={() => onRangeDraft(null)}>Remove selection</button>}{(activeTool === 'tsr' || maintenanceType === 'SPEED_RESTRICTION') && <Stepper label="Speed limit" value={restrictionLimit} min={10} max={160} step={5} suffix=" km/h" onChange={setRestrictionLimit} />}<Stepper label="Duration" value={restrictionDuration} min={10} max={3600} step={10} suffix=" s" onChange={setRestrictionDuration} /><button className="drawer-apply" disabled={disabled || !restrictionTrack || !rangeDraft || rangeDraft.kind !== activeTool || rangeDraft.track_id !== restrictionTrack} onClick={applyRestriction}>Apply {activeTool === 'tsr' ? 'TSR' : maintenanceType === 'FULL_CLOSURE' ? 'full closure' : 'maintenance'}</button></div>}

    {activeTool === 'signal' && <div className="drawer-section"><div className="drawer-section-title"><strong>Manual signal</strong><span>restrictive override only</span></div><label className="drawer-field">Signal<select value={selectedSignalId} onChange={(event) => onSelectedSignal(event.target.value)}>{config.signals.map((signal) => <option key={signal.signal_id} value={signal.signal_id}>{signal.signal_id} · {signal.protected_block_id}</option>)}</select></label><div className="aspect-selector">{(['YELLOW', 'RED'] as SignalAspect[]).map((aspect) => <button key={aspect} type="button" className={`${aspect.toLowerCase()} ${signalAspect === aspect ? 'selected' : ''}`} onClick={() => setSignalAspect(aspect)}>{aspect}</button>)}</div><Stepper label="Duration" value={signalDuration} min={10} max={1800} step={10} suffix=" s" onChange={setSignalDuration} /><button className="drawer-apply" disabled={disabled || !selectedSignalId} onClick={() => onInject({ command: 'inject_signal', signal_id: selectedSignalId, aspect: signalAspect, duration_s: signalDuration })}>Override signal</button><button type="button" className="drawer-remove" disabled={!selectedSignalId} onClick={() => onInject({ command: 'reset_signal', signal_id: selectedSignalId })}>Reset signal to automatic</button></div>}

    <div className="drawer-tip"><strong>Direct manipulation</strong><span>TSR/Maintenance: choose TRACK-UP or TRACK-DOWN, then click only that track for a 200m range and drag either edge. Full closure stops only trains on that selected track.</span></div>
  </aside>
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
  return <section className="panel event-panel"><div className="panel-heading"><h2>Event log</h2><span>{events.length} events</span></div><div className="event-list">{[...events].reverse().map((event) => <div className="event-row" key={event.id}><time>{event.time.toFixed(0)}s</time><span>{event.text}</span></div>)}</section>
}

export default function Dashboard(props: DashboardProps) {
  const { config, frame, frames, selectedTrainId, onSelectTrain, signalStates, crossingStates, constraintState, history, events, playback, exportPaths } = props
  const [constraintsOpen, setConstraintsOpen] = useState(false)
  const [activeTool, setActiveTool] = useState<ConstraintTool>('weather')
  const [selectedSignalId, setSelectedSignalId] = useState(config.signals[0]?.signal_id ?? '')
  const [restrictionTrack, setRestrictionTrack] = useState(config.route.track_ids[0] ?? '')
  const [rangeDraft, setRangeDraft] = useState<RangeDraft | null>(null)
  const [drawerResetKey, setDrawerResetKey] = useState(0)

  const manualSignalOverrides = useMemo(() => Object.fromEntries(constraintState.signals.map((item) => [item.signal_id, { signal_id: item.signal_id, aspect: item.aspect, start_sim_time_s: item.start_time_s ?? frame.sim_time_s, end_sim_time_s: item.end_time_s ?? Number.POSITIVE_INFINITY }])), [constraintState.signals, frame.sim_time_s])

  const resetUi = () => {
    setRangeDraft(null)
    setRestrictionTrack(config.route.track_ids[0] ?? '')
    setActiveTool('weather')
    setDrawerResetKey((value) => value + 1)
  }
  const resetSimulation = () => { resetUi(); props.onReset() }
  const resetConstraints = () => { resetUi(); props.onResetConstraints() }
  const chooseTool = (tool: ConstraintTool) => { if (tool !== activeTool) setRangeDraft(null); setActiveTool(tool) }
  const chooseRestrictionTrack = (trackId: string) => {
    setRestrictionTrack(trackId)
    setRangeDraft((draft) => draft ? { ...draft, track_id: trackId } : draft)
  }

  return <main className={`dashboard ${constraintsOpen ? 'constraints-open' : ''}`}><header className="topbar"><div><span className="eyebrow">Component A · multi-train network simulation</span><h1>Dynamic-ETA Railway Simulator</h1></div><div className="topbar-actions"><button type="button" className={`constraints-toggle ${constraintsOpen ? 'open' : ''}`} onClick={() => setConstraintsOpen((open) => !open)}>Constraints {constraintsOpen ? '×' : '→'}</button><div className="scenario-card"><span>{frame.scenario_id}</span><strong>{frames.length} live trains · {config.stations.length} stations</strong><small>{config.dynamic_signalling ? 'occupancy-driven signalling' : 'scheduled signalling'}</small></div></div></header><div className="dashboard-workspace"><div className="dashboard-main"><SimulationControls {...props} onReset={resetSimulation} onResetConstraints={resetConstraints} /><RailwayRouteV2 config={config} frames={frames} signalStates={signalStates} crossingStates={crossingStates} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} constraintsOpen={constraintsOpen} activeConstraintTool={activeTool} selectedRestrictionTrack={restrictionTrack} rangeDraft={rangeDraft} onRangeDraft={setRangeDraft} onRangeDraftChange={setRangeDraft} onSignalSelect={(signalId) => { setSelectedSignalId(signalId); setActiveTool('signal') }} selectedSignalId={selectedSignalId} manualSignalOverrides={manualSignalOverrides} constraintState={constraintState} /><FleetPanel frames={frames} selectedTrainId={selectedTrainId} onSelectTrain={onSelectTrain} /><div className="two-column"><TrainStatePanel frame={frame} config={config} /><ConstraintPanel frame={frame} /></div><SpeedChart history={history} /><EventLog events={events} /></div>{constraintsOpen && <ConstraintDrawer key={drawerResetKey} config={config} simTime={frame.sim_time_s} playback={playback} activeTool={activeTool} onTool={chooseTool} selectedSignalId={selectedSignalId} onSelectedSignal={setSelectedSignalId} restrictionTrack={restrictionTrack} onRestrictionTrack={chooseRestrictionTrack} rangeDraft={rangeDraft} onRangeDraft={setRangeDraft} onInject={props.onInject} />}</div>{playback.complete && <div className="complete-banner">All trains complete at {frame.sim_time_s.toFixed(0)} s{exportPaths && <div>Dataset exported · CSV: {exportPaths.csv} · Parquet: {exportPaths.parquet}</div>}</div>}</main>
}
