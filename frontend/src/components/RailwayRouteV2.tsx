import { useRef, useState } from 'react'
import type { CrossingState, SignalAspect, SimulatorConfigViz, TelemetryFrame, TrackBlockViz } from '../types'

interface Point { x: number; y: number }
interface Pose extends Point { angleDeg: number }
interface RangeDraft { kind: 'tsr' | 'maintenance'; block_id: string; start_position_m: number; end_position_m: number }
interface ManualSignalOverrideUi { signal_id: string; aspect: SignalAspect; start_sim_time_s: number; end_sim_time_s: number }

type ConstraintTool = 'weather' | 'tsr' | 'maintenance' | 'signal'

const WIDTH = 6000
const HEIGHT = 560
const LEFT = 120
const RIGHT = WIDTH - 120
const BASE_Y = 250
const TRACK_SPACING = 28
const PATH_SAMPLES = 760
const COACH_GAP_PX = 4
const PLATFORM_OFFSET_PX = 34
const STATION_LABEL_OFFSET_PX = 62
const MIN_ZOOM = 0.2
const MAX_ZOOM = 1.25
const ZOOM_STEP = 0.1

function activeAt<T extends { start_time_s: number }>(timeline: T[], simTime: number): T | undefined {
  let active: T | undefined
  for (const entry of timeline) {
    if (entry.start_time_s <= simTime) active = entry
    else break
  }
  return active
}

function curveDelta(block: TrackBlockViz): number {
  if (!block.curve_radius_m || !block.curve_direction) return 0
  const severity = Math.max(28, Math.min(70, 52000 / block.curve_radius_m))
  return block.curve_direction === 'RIGHT' ? severity : -severity
}

function geometry(config: SimulatorConfigViz) {
  const total = config.route.total_length_m
  let y = BASE_Y
  return config.route.blocks.map((block) => {
    const x0 = LEFT + (block.route_start_m / total) * (RIGHT - LEFT)
    const x1 = LEFT + (block.route_end_m / total) * (RIGHT - LEFT)
    const y0 = y
    const y1 = y + curveDelta(block)
    y = y1
    return { block, x0, x1, y0, y1 }
  })
}

function metersToPixels(config: SimulatorConfigViz, meters: number): number {
  return (meters / Math.max(1, config.route.total_length_m)) * (RIGHT - LEFT)
}

function trainCompartmentCount(lengthM: number): number {
  if (lengthM >= 300) return 3
  if (lengthM >= 280) return 2
  return 1
}

function bezierPoint(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u = 1 - t
  const uu = u * u
  const tt = t * t
  return { x: uu * u * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + tt * t * p3.x, y: uu * u * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + tt * t * p3.y }
}

function bezierTangent(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u = 1 - t
  return { x: 3 * u * u * (p1.x - p0.x) + 6 * u * t * (p2.x - p1.x) + 3 * t * t * (p3.x - p2.x), y: 3 * u * u * (p1.y - p0.y) + 6 * u * t * (p2.y - p1.y) + 3 * t * t * (p3.y - p2.y) }
}

function centerPoseAt(config: SimulatorConfigViz, routePositionM: number): Pose {
  const blocks = geometry(config)
  const clamped = Math.max(0, Math.min(routePositionM, config.route.total_length_m))
  const item = blocks.find(({ block }) => clamped <= block.route_end_m) ?? blocks[blocks.length - 1]
  const t = Math.max(0, Math.min(1, (clamped - item.block.route_start_m) / Math.max(1, item.block.length_m)))
  const p0 = { x: item.x0, y: item.y0 }
  const p3 = { x: item.x1, y: item.y1 }
  let point: Point
  let tangent: Point
  if (item.block.curve_direction && item.block.curve_radius_m) {
    const dx = item.x1 - item.x0
    const p1 = { x: item.x0 + dx * .34, y: item.y0 }
    const p2 = { x: item.x0 + dx * .66, y: item.y1 }
    point = bezierPoint(p0, p1, p2, p3, t)
    tangent = bezierTangent(p0, p1, p2, p3, t)
  } else {
    point = { x: item.x0 + (item.x1 - item.x0) * t, y: item.y0 + (item.y1 - item.y0) * t }
    tangent = { x: item.x1 - item.x0, y: item.y1 - item.y0 }
  }
  const angleDeg = Math.atan2(tangent.y, tangent.x) * 180 / Math.PI
  if (routePositionM === clamped) return { ...point, angleDeg }
  const extensionPx = metersToPixels(config, routePositionM - clamped)
  const angle = angleDeg * Math.PI / 180
  return { x: point.x + Math.cos(angle) * extensionPx, y: point.y + Math.sin(angle) * extensionPx, angleDeg }
}

function poseAt(config: SimulatorConfigViz, routePositionM: number, trackIdx: number): Pose {
  const center = centerPoseAt(config, routePositionM)
  const angle = center.angleDeg * Math.PI / 180
  const offset = (trackIdx - (config.route.track_ids.length - 1) / 2) * TRACK_SPACING
  return { x: center.x - Math.sin(angle) * offset, y: center.y + Math.cos(angle) * offset, angleDeg: center.angleDeg }
}

function offsetPose(pose: Pose, lateralOffsetPx: number): Point {
  const angle = pose.angleDeg * Math.PI / 180
  return { x: pose.x - Math.sin(angle) * lateralOffsetPx, y: pose.y + Math.cos(angle) * lateralOffsetPx }
}

function trackIndex(config: SimulatorConfigViz, trackId: string): number {
  const i = config.route.track_ids.indexOf(trackId)
  return i >= 0 ? i : 0
}

function pointAt(config: SimulatorConfigViz, routePositionM: number, trackIdx: number): Point {
  const { x, y } = poseAt(config, routePositionM, trackIdx)
  return { x, y }
}

function pathBetween(config: SimulatorConfigViz, startM: number, endM: number, trackIdx: number, samples = PATH_SAMPLES): string {
  const span = Math.max(1, endM - startM)
  const count = Math.max(8, Math.ceil((span / config.route.total_length_m) * samples))
  const points: Point[] = []
  for (let i = 0; i <= count; i += 1) points.push(pointAt(config, startM + span * i / count, trackIdx))
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ')
}

function offsetPathBetween(config: SimulatorConfigViz, startM: number, endM: number, trackIdx: number, lateralOffsetPx: number): string {
  const span = Math.max(1, endM - startM)
  const count = Math.max(10, Math.ceil((span / config.route.total_length_m) * PATH_SAMPLES))
  const points: Point[] = []
  for (let i = 0; i <= count; i += 1) {
    const routePositionM = startM + span * i / count
    points.push(offsetPose(poseAt(config, routePositionM, trackIdx), lateralOffsetPx))
  }
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ')
}

function fullTrackPath(config: SimulatorConfigViz, trackIdx: number): string { return pathBetween(config, 0, config.route.total_length_m, trackIdx) }

function signalFallback(config: SimulatorConfigViz, signalId: string, simTime: number): SignalAspect {
  const schedule = config.environment.signal_states.find((item) => item.signal_id === signalId)
  return activeAt(schedule?.timeline ?? [], simTime)?.aspect ?? 'GREEN'
}

function displayPoseAt(config: SimulatorConfigViz, frame: TelemetryFrame, routePositionM: number): Pose {
  const run = config.trains.find((item) => item.train.train_id === frame.train_id)
  const plan = run?.track_changes[0]
  const crossover = plan ? config.crossovers.find((item) => item.crossover_id === plan.crossover_id) : undefined
  const hasCompletedTurnaround = Boolean(plan?.reverse_after_change && frame.direction === 'REVERSE' && frame.track_id === crossover?.to_track_id)
  if (crossover && !hasCompletedTurnaround) {
    if (routePositionM >= crossover.route_start_m && routePositionM <= crossover.route_end_m) {
      const t = (routePositionM - crossover.route_start_m) / Math.max(1, crossover.route_end_m - crossover.route_start_m)
      const from = poseAt(config, routePositionM, trackIndex(config, crossover.from_track_id))
      const to = poseAt(config, routePositionM, trackIndex(config, crossover.to_track_id))
      return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t, angleDeg: from.angleDeg }
    }
    if (frame.track_id === crossover.to_track_id && routePositionM < crossover.route_start_m) return poseAt(config, routePositionM, trackIndex(config, crossover.from_track_id))
    if (frame.track_id === crossover.from_track_id && routePositionM > crossover.route_end_m) return poseAt(config, routePositionM, trackIndex(config, crossover.to_track_id))
  }
  return poseAt(config, routePositionM, trackIndex(config, frame.track_id))
}

function clampZoom(value: number): number { return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value)) }
function snapHalf(value: number): number { return Math.round(value * 2) / 2 }
function remaining(end: number | null, simTime: number): string { return end == null ? 'until reset' : `${Math.max(0, Math.ceil(end - simTime))}s left` }

export default function RailwayRouteV2({ config, frames, signalStates, crossingStates, selectedTrainId, onSelectTrain, constraintsOpen = false, activeConstraintTool = 'weather', rangeDraft = null, onRangeDraft, onRangeDraftChange, onSignalSelect, selectedSignalId = '', manualSignalOverrides = {} }: {
  config: SimulatorConfigViz
  frames: TelemetryFrame[]
  signalStates: Record<string, SignalAspect>
  crossingStates: Record<string, CrossingState>
  selectedTrainId: string
  onSelectTrain: (trainId: string) => void
  constraintsOpen?: boolean
  activeConstraintTool?: ConstraintTool
  rangeDraft?: RangeDraft | null
  onRangeDraft?: (draft: RangeDraft) => void
  onRangeDraftChange?: (draft: RangeDraft | null) => void
  onSignalSelect?: (signalId: string) => void
  selectedSignalId?: string
  manualSignalOverrides?: Record<string, ManualSignalOverrideUi>
}) {
  const simTime = frames[0]?.sim_time_s ?? 0
  const blocks = geometry(config)
  const scrollRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const dragHandle = useRef<'start' | 'end' | null>(null)
  const [zoom, setZoom] = useState(0.55)

  const fitRoute = () => {
    const viewportWidth = scrollRef.current?.clientWidth ?? WIDTH
    setZoom(clampZoom((viewportWidth - 28) / WIDTH))
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
  }

  const pointerRoutePosition = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return 0
    const svgX = (clientX - rect.left) * WIDTH / Math.max(1, rect.width)
    const ratio = Math.max(0, Math.min(1, (svgX - LEFT) / (RIGHT - LEFT)))
    return ratio * config.route.total_length_m
  }

  const chooseRange = (clientX: number) => {
    if (!constraintsOpen || (activeConstraintTool !== 'tsr' && activeConstraintTool !== 'maintenance')) return
    const routeM = pointerRoutePosition(clientX)
    const block = config.route.blocks.find((item) => routeM >= item.route_start_m && routeM <= item.route_end_m) ?? config.route.blocks.at(-1)
    if (!block) return
    const local = routeM - block.route_start_m
    const width = Math.min(200, block.length_m)
    let start = local - width / 2
    let end = local + width / 2
    if (start < 0) { end -= start; start = 0 }
    if (end > block.length_m) { start -= end - block.length_m; end = block.length_m }
    start = Math.max(0, start)
    end = Math.min(block.length_m, end)
    onRangeDraft?.({ kind: activeConstraintTool, block_id: block.block_id, start_position_m: Math.round(start), end_position_m: Math.round(end) })
  }

  const dragRange = (clientX: number) => {
    if (!dragHandle.current || !rangeDraft || !onRangeDraftChange) return
    const block = config.route.blocks.find((item) => item.block_id === rangeDraft.block_id)
    if (!block) return
    const local = Math.round((pointerRoutePosition(clientX) - block.route_start_m) / 10) * 10
    if (dragHandle.current === 'start') onRangeDraftChange({ ...rangeDraft, start_position_m: Math.max(0, Math.min(local, rangeDraft.end_position_m - 10)) })
    else onRangeDraftChange({ ...rangeDraft, end_position_m: Math.min(block.length_m, Math.max(local, rangeDraft.start_position_m + 10)) })
  }

  const activeTsr = config.environment.temporary_speed_restrictions.filter((item) => item.start_time_s <= simTime && (item.end_time_s == null || simTime < item.end_time_s))
  const activeMaintenance = config.environment.maintenance_restrictions.filter((item) => item.start_time_s <= simTime && (item.end_time_s == null || simTime < item.end_time_s))

  return <section className={`panel route-panel realistic-route-panel route-focus ${constraintsOpen ? 'constraint-editing' : ''}`}>
    <div className="panel-heading route-panel-heading"><div><span className="eyebrow">Live railway network</span><h2>{config.route.route_name}</h2></div><div className="route-heading-tools"><div className="route-meta">{config.route.track_ids.length} tracks · {frames.length} trains · {config.stations.length} stations{constraintsOpen && <b className="edit-mode-pill">EDIT MODE</b>}</div><div className="route-zoom-controls"><button type="button" onClick={() => setZoom((value) => clampZoom(value - ZOOM_STEP))}>−</button><button type="button" onClick={fitRoute}>Fit</button><button type="button" onClick={() => setZoom((value) => clampZoom(value + ZOOM_STEP))}>+</button><button type="button" className="zoom-value" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button></div></div></div>

    <div className="realistic-route-scroll" ref={scrollRef}><svg ref={svgRef} className="realistic-route" style={{ width: `${WIDTH * zoom}px` }} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} onPointerMove={(event) => dragRange(event.clientX)} onPointerUp={() => { dragHandle.current = null }} onPointerLeave={() => { dragHandle.current = null }}>
      {blocks.slice(1).map(({ block }) => { const p = pointAt(config, block.route_start_m, 0); return <line key={block.block_id} x1={p.x} y1={55} x2={p.x} y2={HEIGHT - 55} className="block-boundary" /> })}

      {config.route.track_ids.map((id, index) => <g key={id}><path d={fullTrackPath(config, index)} className="track-ballast-path" /><path d={fullTrackPath(config, index)} className="track-bed-path" /><path d={fullTrackPath(config, index)} className="track-sleeper-path" /><path d={fullTrackPath(config, index)} className="track-rail-path" /><text x={24} y={pointAt(config, 0, index).y + 4} className="track-name">{id}</text></g>)}

      {constraintsOpen && (activeConstraintTool === 'tsr' || activeConstraintTool === 'maintenance') && config.route.track_ids.map((_, trackIdx) => blocks.map(({ block }) => <path key={`hit-${trackIdx}-${block.block_id}`} d={pathBetween(config, block.route_start_m, block.route_end_m, trackIdx)} className="constraint-hit-path" onClick={(event) => { event.stopPropagation(); chooseRange(event.clientX) }} />))}

      {config.crossovers.map((xover) => { const fromIdx = trackIndex(config, xover.from_track_id); const toIdx = trackIndex(config, xover.to_track_id); const a = pointAt(config, xover.route_start_m, fromIdx); const b = pointAt(config, xover.route_end_m, toIdx); const c = pointAt(config, xover.route_start_m, toIdx); const d = pointAt(config, xover.route_end_m, fromIdx); return <g key={xover.crossover_id} className="svg-crossover"><path d={`M ${a.x} ${a.y} L ${b.x} ${b.y}`} /><path d={`M ${c.x} ${c.y} L ${d.x} ${d.y}`} /><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 12} textAnchor="middle">⇄ {xover.crossover_id}</text></g> })}

      {config.stations.map((station) => { const labelPlatform = station.platforms[0]; const labelTrackIdx = trackIndex(config, labelPlatform.track_id); const labelPose = poseAt(config, labelPlatform.route_position_m, labelTrackIdx); const labelPoint = offsetPose(labelPose, -STATION_LABEL_OFFSET_PX); return <g key={station.station_id} className="svg-station">{station.platforms.map((platform) => { const idx = trackIndex(config, platform.track_id); const side = idx === 0 ? -1 : 1; const d = offsetPathBetween(config, platform.route_position_m - platform.length_m / 2, platform.route_position_m + platform.length_m / 2, idx, side * PLATFORM_OFFSET_PX); return <g key={platform.platform_id} className="svg-platform"><path d={d} className="platform-surface" /><path d={d} className="platform-edge" /></g> })}<text x={labelPoint.x} y={labelPoint.y} textAnchor="middle" className="station-label">▰ {station.station_name}</text></g> })}

      {blocks.map(({ block }) => {
        const centerM = block.route_start_m + block.length_m / 2
        const p = pointAt(config, centerM, 0)
        const labelY = Math.max(24, p.y - 94)
        const weatherSchedule = config.environment.weather.find((item) => item.block_id === block.block_id)
        const weather = activeAt(weatherSchedule?.timeline ?? [], simTime)
        const tsrs = activeTsr.filter((item) => item.block_id === block.block_id)
        const maintenance = activeMaintenance.filter((item) => item.block_id === block.block_id)
        const manualSignals = config.signals.filter((signal) => signal.protected_block_id === block.block_id && manualSignalOverrides[signal.signal_id])
        const lines = [weather && weather.condition !== 'CLEAR' ? `${weather.condition.replaceAll('_', ' ')} · ${weather.visibility_m}m` : null, ...tsrs.map((item) => `TSR ${item.speed_limit_kmh} · ${remaining(item.end_time_s, simTime)}`), ...maintenance.map((item) => `MAINT ${item.speed_limit_kmh} · ${remaining(item.end_time_s, simTime)}`), ...manualSignals.map((signal) => `MANUAL ${signal.signal_id} ${manualSignalOverrides[signal.signal_id].aspect} · ${remaining(manualSignalOverrides[signal.signal_id].end_sim_time_s, simTime)}`)].filter(Boolean) as string[]
        return <g key={block.block_id} className="block-svg-label"><text x={p.x} y={labelY} textAnchor="middle">{block.block_id}</text><text x={p.x} y={labelY + 16} textAnchor="middle" className="curve-label">MAX {block.speed_limit_kmh} km/h</text>{lines.slice(0, 4).map((line, i) => <g key={line}><rect x={p.x - 90} y={p.y + 62 + i * 24} width="180" height="19" rx="7" className="block-constraint-bg" /><text x={p.x} y={p.y + 76 + i * 24} textAnchor="middle" className="block-constraint-text">{line}</text></g>)}</g>
      })}

      {activeTsr.map((item) => config.route.track_ids.map((trackId, trackIdx) => <g key={`${item.restriction_id}-${trackId}`}><path d={pathBetween(config, item.route_start_m, item.route_end_m, trackIdx)} className="restriction-line tsr-line" /><text x={pointAt(config, (item.route_start_m + item.route_end_m) / 2, trackIdx).x} y={pointAt(config, (item.route_start_m + item.route_end_m) / 2, trackIdx).y - 18} textAnchor="middle" className="restriction-track-label">TSR · {item.speed_limit_kmh} km/h · {remaining(item.end_time_s, simTime)}</text></g>))}
      {activeMaintenance.map((item) => config.route.track_ids.map((trackId, trackIdx) => <g key={`${item.restriction_id}-${trackId}`}><path d={pathBetween(config, item.route_start_m, item.route_end_m, trackIdx)} className="restriction-line maintenance-line" /><text x={pointAt(config, (item.route_start_m + item.route_end_m) / 2, trackIdx).x} y={pointAt(config, (item.route_start_m + item.route_end_m) / 2, trackIdx).y + 28} textAnchor="middle" className="restriction-track-label maintenance-label">MAINT · {item.speed_limit_kmh} km/h · {remaining(item.end_time_s, simTime)}</text></g>))}

      {rangeDraft && constraintsOpen && (activeConstraintTool === 'tsr' || activeConstraintTool === 'maintenance') && (() => { const block = config.route.blocks.find((item) => item.block_id === rangeDraft.block_id); if (!block) return null; const startRoute = block.route_start_m + rangeDraft.start_position_m; const endRoute = block.route_start_m + rangeDraft.end_position_m; return <g className={`range-preview ${rangeDraft.kind}`}>{config.route.track_ids.map((_, idx) => <path key={idx} d={pathBetween(config, startRoute, endRoute, idx)} className="range-preview-line" />)}{[startRoute, endRoute].map((routeM, index) => { const p = pointAt(config, routeM, 0); return <g key={index} className="range-handle" transform={`translate(${p.x} ${p.y})`} onPointerDown={(event) => { event.stopPropagation(); dragHandle.current = index === 0 ? 'start' : 'end'; (event.currentTarget as SVGGElement).setPointerCapture?.(event.pointerId) }}><circle r="13" /><line x1="0" y1="-22" x2="0" y2="22" /></g> })}</g> })()}

      {config.signals.map((signal) => { const idx = trackIndex(config, signal.track_id); const p = poseAt(config, signal.route_position_m, idx); const aspect = signalStates[signal.signal_id] ?? signalFallback(config, signal.signal_id, simTime); const side = signal.direction === 'FORWARD' ? -1 : 1; const labelY = side * 47; const manual = manualSignalOverrides[signal.signal_id]; const selected = constraintsOpen && selectedSignalId === signal.signal_id; return <g key={signal.signal_id} transform={`translate(${p.x} ${p.y}) rotate(${p.angleDeg})`} className={`svg-signal ${constraintsOpen ? 'clickable' : ''} ${selected ? 'selected-signal' : ''} ${manual ? 'manual-signal' : ''}`} onClick={(event) => { if (constraintsOpen) { event.stopPropagation(); onSignalSelect?.(signal.signal_id) } }}><line x1="0" y1="0" x2="0" y2={side * 28} /><circle cx="0" cy={side * 34} r="7" className={`svg-signal-light ${aspect.toLowerCase()}`} />{manual && <circle cx="13" cy={side * 34} r="6" className="manual-marker" />}<text x="0" y={labelY} textAnchor="middle" transform={`rotate(${-p.angleDeg} 0 ${labelY})`}>{signal.signal_id}{manual ? ` · MANUAL ${remaining(manual.end_sim_time_s, simTime)}` : ''}</text></g> })}

      {config.environment.crossings.map((crossing, crossingIndex) => { const fallback = activeAt(crossing.timeline, simTime)?.state ?? 'CLOSED_FOR_TRAIN'; const state = crossingStates[crossing.crossing_id] ?? fallback; const trainMustStop = state === 'CLOSED_FOR_TRAIN'; const center = centerPoseAt(config, crossing.route_position_m); const visualClass = trainMustStop ? 'road-open' : 'road-closed'; const statusText = trainMustStop ? 'ROAD OPEN' : 'ROAD CLOSED'; const labelY = crossing.crossing_id === 'XING-001' ? 78 : (crossingIndex % 2 === 0 ? -68 : 78); return <g key={crossing.crossing_id} transform={`translate(${center.x} ${center.y}) rotate(${center.angleDeg})`} className={`svg-crossing ${visualClass}`}><line className="crossing-road-bed" x1="0" y1="-54" x2="0" y2="54" /><line className="crossing-road-mark" x1="0" y1="-54" x2="0" y2="54" /><line className="crossing-gate" x1="-18" y1="-38" x2="18" y2="-38" /><line className="crossing-gate" x1="-18" y1="38" x2="18" y2="38" /><text x="0" y={labelY} textAnchor="middle" transform={`rotate(${-center.angleDeg} 0 ${labelY})`}>{crossing.crossing_id} · {statusText}</text></g> })}

      {frames.map((frame, index) => { const reverse = frame.direction === 'REVERSE'; const directionSign = reverse ? -1 : 1; const selected = frame.train_id === selectedTrainId; const run = config.trains.find((item) => item.train.train_id === frame.train_id); const trainLengthM = run?.train.length_m ?? config.train.length_m; const compartmentCount = trainCompartmentCount(trainLengthM); const compartmentLengthM = trainLengthM / compartmentCount; const coachWidthPx = Math.max(26, metersToPixels(config, compartmentLengthM) - COACH_GAP_PX); const frontPose = displayPoseAt(config, frame, frame.route_position_m); const labelX = snapHalf(frontPose.x); const labelY = snapHalf(frontPose.y - 38); return <g key={frame.train_id} className={`svg-train-position train-${index % 4} ${frame.active ? 'active' : 'waiting'} ${frame.completed ? 'completed' : ''} ${selected ? 'selected' : ''}`} onClick={() => onSelectTrain(frame.train_id)}>{Array.from({ length: compartmentCount }, (_, compartmentIndex) => { const distanceBehindFrontM = (compartmentIndex + 0.5) * compartmentLengthM; const coachRoutePositionM = frame.route_position_m - directionSign * distanceBehindFrontM; const coachPose = displayPoseAt(config, frame, coachRoutePositionM); const isFront = compartmentIndex === 0; const x = -coachWidthPx / 2; return <g key={compartmentIndex} transform={`translate(${coachPose.x} ${coachPose.y}) rotate(${coachPose.angleDeg}) ${reverse ? 'scale(-1 1)' : ''}`} className="svg-train-body train-compartment"><rect x={x} y="-12" width={coachWidthPx} height="20" rx="5" className="train-coach-shell" /><rect x={x + coachWidthPx * .2} y="-8" width={Math.max(5, coachWidthPx * .17)} height="6" rx="1.5" className="train-window" /><rect x={x + coachWidthPx * .5} y="-8" width={Math.max(5, coachWidthPx * .17)} height="6" rx="1.5" className="train-window" />{isFront && <path d={`M ${coachWidthPx / 2 - 9} -12 L ${coachWidthPx / 2} -6 L ${coachWidthPx / 2} 5 L ${coachWidthPx / 2 - 9} 8 Z`} className="train-cab" />}<circle cx={x + 9} cy="10" r="3" /><circle cx={x + coachWidthPx - 9} cy="10" r="3" /></g> })}<g transform={`translate(${labelX} ${labelY})`} className="train-label-anchor"><rect x="-72" y="-12" width="144" height="20" rx="8" className="train-label-bg" /><text x="0" y="-1" textAnchor="middle" dominantBaseline="middle" className="train-speed-label">{frame.train_id} · {Math.round(frame.speed_kmh)}</text></g></g> })}

      <text x={LEFT} y={HEIGHT - 16} className="endpoint-svg-label">DELHI SIDE</text><text x={RIGHT} y={HEIGHT - 16} textAnchor="end" className="endpoint-svg-label">AGRA SIDE</text>
    </svg></div>
    <div className="legend"><span>dynamic signals = block occupancy</span><span>click track in edit mode = 200m draft</span><span>drag handles = resize range</span><span className="crossing-legend road-open-legend">● road open = train stop</span><span><i className="legend-swatch tsr-swatch" />TSR</span><span><i className="legend-swatch maintenance-swatch" />Maintenance</span></div>
  </section>
}
