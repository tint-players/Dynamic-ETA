import { useRef, useState } from 'react'
import type { CrossingState, SignalAspect, SimulatorConfigViz, TelemetryFrame, TrackBlockViz } from '../types'

interface Point { x: number; y: number }
interface Pose extends Point { angleDeg: number }

const WIDTH = 6000
const HEIGHT = 430
const LEFT = 120
const RIGHT = WIDTH - 120
const BASE_Y = 220
const TRACK_SPACING = 24
const PATH_SAMPLES = 760
const COACH_GAP_PX = 4
const PLATFORM_OFFSET_PX = 31
const STATION_LABEL_OFFSET_PX = 55
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
  return {
    x: uu * u * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + tt * t * p3.x,
    y: uu * u * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + tt * t * p3.y,
  }
}

function bezierTangent(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u = 1 - t
  return {
    x: 3 * u * u * (p1.x - p0.x) + 6 * u * t * (p2.x - p1.x) + 3 * t * t * (p3.x - p2.x),
    y: 3 * u * u * (p1.y - p0.y) + 6 * u * t * (p2.y - p1.y) + 3 * t * t * (p3.y - p2.y),
  }
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
  return {
    x: point.x + Math.cos(angle) * extensionPx,
    y: point.y + Math.sin(angle) * extensionPx,
    angleDeg,
  }
}

function poseAt(config: SimulatorConfigViz, routePositionM: number, trackIdx: number): Pose {
  const center = centerPoseAt(config, routePositionM)
  const angle = center.angleDeg * Math.PI / 180
  const offset = (trackIdx - (config.route.track_ids.length - 1) / 2) * TRACK_SPACING
  return {
    x: center.x - Math.sin(angle) * offset,
    y: center.y + Math.cos(angle) * offset,
    angleDeg: center.angleDeg,
  }
}

function offsetPose(pose: Pose, lateralOffsetPx: number): Point {
  const angle = pose.angleDeg * Math.PI / 180
  return {
    x: pose.x - Math.sin(angle) * lateralOffsetPx,
    y: pose.y + Math.cos(angle) * lateralOffsetPx,
  }
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

function fullTrackPath(config: SimulatorConfigViz, trackIdx: number): string {
  return pathBetween(config, 0, config.route.total_length_m, trackIdx)
}

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
      return {
        x: from.x + (to.x - from.x) * t,
        y: from.y + (to.y - from.y) * t,
        angleDeg: from.angleDeg,
      }
    }

    if (frame.track_id === crossover.to_track_id && routePositionM < crossover.route_start_m) {
      return poseAt(config, routePositionM, trackIndex(config, crossover.from_track_id))
    }
    if (frame.track_id === crossover.from_track_id && routePositionM > crossover.route_end_m) {
      return poseAt(config, routePositionM, trackIndex(config, crossover.to_track_id))
    }
  }

  return poseAt(config, routePositionM, trackIndex(config, frame.track_id))
}

function clampZoom(value: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value))
}

function snapHalf(value: number): number {
  return Math.round(value * 2) / 2
}

export default function RailwayRouteV2({ config, frames, signalStates, crossingStates, selectedTrainId, onSelectTrain }: {
  config: SimulatorConfigViz
  frames: TelemetryFrame[]
  signalStates: Record<string, SignalAspect>
  crossingStates: Record<string, CrossingState>
  selectedTrainId: string
  onSelectTrain: (trainId: string) => void
}) {
  const simTime = frames[0]?.sim_time_s ?? 0
  const blocks = geometry(config)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(0.55)

  const fitRoute = () => {
    const viewportWidth = scrollRef.current?.clientWidth ?? WIDTH
    const nextZoom = clampZoom((viewportWidth - 28) / WIDTH)
    setZoom(nextZoom)
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
  }

  const resetZoom = () => setZoom(1)

  return (
    <section className="panel route-panel realistic-route-panel">
      <div className="panel-heading route-panel-heading">
        <div><span className="eyebrow">Live railway network</span><h2>{config.route.route_name}</h2></div>
        <div className="route-heading-tools">
          <div className="route-meta">{config.route.track_ids.length} tracks · {frames.length} trains · {config.stations.length} stations</div>
          <div className="route-zoom-controls" aria-label="Route zoom controls">
            <button type="button" onClick={() => setZoom((value) => clampZoom(value - ZOOM_STEP))} aria-label="Zoom out">−</button>
            <button type="button" onClick={fitRoute}>Fit</button>
            <button type="button" onClick={() => setZoom((value) => clampZoom(value + ZOOM_STEP))} aria-label="Zoom in">+</button>
            <button type="button" className="zoom-value" onClick={resetZoom} title="Reset to 100%">{Math.round(zoom * 100)}%</button>
          </div>
        </div>
      </div>

      <div className="realistic-route-scroll" ref={scrollRef}>
        <svg className="realistic-route" style={{ width: `${WIDTH * zoom}px` }} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Multi-train curved railway network">
          {blocks.slice(1).map(({ block }) => {
            const p = pointAt(config, block.route_start_m, 0)
            return <line key={block.block_id} x1={p.x} y1={50} x2={p.x} y2={HEIGHT - 48} className="block-boundary" />
          })}

          {config.route.track_ids.map((id, index) => (
            <g key={id}>
              <path d={fullTrackPath(config, index)} className="track-ballast-path" />
              <path d={fullTrackPath(config, index)} className="track-bed-path" />
              <path d={fullTrackPath(config, index)} className="track-sleeper-path" />
              <path d={fullTrackPath(config, index)} className="track-rail-path" />
              <text x={24} y={pointAt(config, 0, index).y + 4} className="track-name">{id}</text>
            </g>
          ))}

          {config.crossovers.map((xover) => {
            const fromIdx = trackIndex(config, xover.from_track_id)
            const toIdx = trackIndex(config, xover.to_track_id)
            const a = pointAt(config, xover.route_start_m, fromIdx)
            const b = pointAt(config, xover.route_end_m, toIdx)
            const c = pointAt(config, xover.route_start_m, toIdx)
            const d = pointAt(config, xover.route_end_m, fromIdx)
            return <g key={xover.crossover_id} className="svg-crossover"><path d={`M ${a.x} ${a.y} L ${b.x} ${b.y}`} /><path d={`M ${c.x} ${c.y} L ${d.x} ${d.y}`} /><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 12} textAnchor="middle">⇄ {xover.crossover_id}</text></g>
          })}

          {config.stations.map((station) => {
            const labelPlatform = station.platforms[0]
            const labelTrackIdx = trackIndex(config, labelPlatform.track_id)
            const labelPose = poseAt(config, labelPlatform.route_position_m, labelTrackIdx)
            const labelPoint = offsetPose(labelPose, -STATION_LABEL_OFFSET_PX)

            return <g key={station.station_id} className="svg-station">
              {station.platforms.map((platform) => {
                const idx = trackIndex(config, platform.track_id)
                const side = idx === 0 ? -1 : 1
                const halfLengthM = platform.length_m / 2
                const startM = platform.route_position_m - halfLengthM
                const endM = platform.route_position_m + halfLengthM
                const d = offsetPathBetween(config, startM, endM, idx, side * PLATFORM_OFFSET_PX)
                return <g key={platform.platform_id} className="svg-platform">
                  <path d={d} className="platform-surface" />
                  <path d={d} className="platform-edge" />
                </g>
              })}
              <text x={labelPoint.x} y={labelPoint.y} textAnchor="middle" className="station-label">▰ {station.station_name}</text>
            </g>
          })}

          {blocks.map(({ block }) => {
            const p = pointAt(config, block.route_start_m + block.length_m / 2, 0)
            const labelLift = block.block_id === 'BLK-03' ? 18 : 0
            const blockLabelY = Math.max(22, p.y - 82 - labelLift)
            const speedLabelY = Math.max(37, p.y - 67 - labelLift)
            const curveLabelY = Math.max(52, p.y - 52 - labelLift)
            return <g key={block.block_id} className="block-svg-label"><text x={p.x} y={blockLabelY} textAnchor="middle">{block.block_id}</text><text x={p.x} y={speedLabelY} textAnchor="middle" className="curve-label">MAX {block.speed_limit_kmh} km/h</text>{block.curve_radius_m && <text x={p.x} y={curveLabelY} textAnchor="middle" className="curve-label">{block.curve_direction === 'LEFT' ? '↶' : '↷'} R{block.curve_radius_m}m</text>}</g>
          })}

          {config.environment.temporary_speed_restrictions.map((item) => config.route.track_ids.map((trackId, trackIdx) => <path key={`${item.restriction_id}-${trackId}`} d={pathBetween(config, item.route_start_m, item.route_end_m, trackIdx)} className="restriction-line tsr-line" />))}
          {config.environment.maintenance_restrictions.map((item) => config.route.track_ids.map((trackId, trackIdx) => <path key={`${item.restriction_id}-${trackId}`} d={pathBetween(config, item.route_start_m, item.route_end_m, trackIdx)} className="restriction-line maintenance-line" />))}

          {config.signals.map((signal) => {
            const idx = trackIndex(config, signal.track_id)
            const p = poseAt(config, signal.route_position_m, idx)
            const aspect = signalStates[signal.signal_id] ?? signalFallback(config, signal.signal_id, simTime)
            const side = signal.direction === 'FORWARD' ? -1 : 1
            const labelY = side * 47
            return <g key={signal.signal_id} transform={`translate(${p.x} ${p.y}) rotate(${p.angleDeg})`} className="svg-signal"><line x1="0" y1="0" x2="0" y2={side * 28} /><circle cx="0" cy={side * 34} r="7" className={`svg-signal-light ${aspect.toLowerCase()}`} /><text x="0" y={labelY} textAnchor="middle" transform={`rotate(${-p.angleDeg} 0 ${labelY})`}>{signal.signal_id}</text></g>
          })}

          {config.environment.crossings.map((crossing, crossingIndex) => {
            const fallback = activeAt(crossing.timeline, simTime)?.state ?? 'CLOSED_FOR_TRAIN'
            const state = crossingStates[crossing.crossing_id] ?? fallback
            const trainMustStop = state === 'CLOSED_FOR_TRAIN'
            const center = centerPoseAt(config, crossing.route_position_m)
            const visualClass = trainMustStop ? 'road-open' : 'road-closed'
            const statusText = trainMustStop ? 'ROAD OPEN' : 'ROAD CLOSED'
            const railText = trainMustStop ? 'TRAIN STOP' : 'RAIL PROTECTED'
            const labelY = crossing.crossing_id === 'XING-001' ? 78 : (crossingIndex % 2 === 0 ? -68 : 78)
            return <g key={crossing.crossing_id} transform={`translate(${center.x} ${center.y}) rotate(${center.angleDeg})`} className={`svg-crossing ${visualClass}`}><line className="crossing-road-bed" x1="0" y1="-54" x2="0" y2="54" /><line className="crossing-road-mark" x1="0" y1="-54" x2="0" y2="54" /><line className="crossing-gate" x1="-18" y1="-38" x2="18" y2="-38" /><line className="crossing-gate" x1="-18" y1="38" x2="18" y2="38" /><text x="0" y={labelY} textAnchor="middle" transform={`rotate(${-center.angleDeg} 0 ${labelY})`}><tspan x="0" dy="0">{crossing.crossing_id} · {statusText}</tspan><tspan x="0" dy="13">{railText}</tspan></text></g>
          })}

          {frames.map((frame, index) => {
            const reverse = frame.direction === 'REVERSE'
            const directionSign = reverse ? -1 : 1
            const selected = frame.train_id === selectedTrainId
            const run = config.trains.find((item) => item.train.train_id === frame.train_id)
            const trainLengthM = run?.train.length_m ?? config.train.length_m
            const compartmentCount = trainCompartmentCount(trainLengthM)
            const compartmentLengthM = trainLengthM / compartmentCount
            const coachWidthPx = Math.max(26, metersToPixels(config, compartmentLengthM) - COACH_GAP_PX)
            const frontPose = displayPoseAt(config, frame, frame.route_position_m)
            const labelX = snapHalf(frontPose.x)
            const labelY = snapHalf(frontPose.y - 34)

            return <g key={frame.train_id} className={`svg-train-position train-${index % 4} ${frame.active ? 'active' : 'waiting'} ${frame.completed ? 'completed' : ''} ${selected ? 'selected' : ''}`} onClick={() => onSelectTrain(frame.train_id)}>
              {Array.from({ length: compartmentCount }, (_, compartmentIndex) => {
                const distanceBehindFrontM = (compartmentIndex + 0.5) * compartmentLengthM
                const coachRoutePositionM = frame.route_position_m - directionSign * distanceBehindFrontM
                const coachPose = displayPoseAt(config, frame, coachRoutePositionM)
                const isFront = compartmentIndex === 0
                const isRear = compartmentIndex === compartmentCount - 1
                const x = -coachWidthPx / 2

                return <g key={compartmentIndex} transform={`translate(${coachPose.x} ${coachPose.y}) rotate(${coachPose.angleDeg}) ${reverse ? 'scale(-1 1)' : ''}`} className={`svg-train-body train-compartment ${isFront ? 'front-coach' : ''} ${isRear ? 'rear-coach' : ''}`}>
                  <rect x={x} y="-12" width={coachWidthPx} height="20" rx="5" className="train-coach-shell" />
                  <line x1={x + 4} y1="-10" x2={x + coachWidthPx - 4} y2="-10" className="train-roof-line" />
                  <rect x={x + coachWidthPx * .18} y="-8" width={Math.max(5, coachWidthPx * .16)} height="6" rx="1.5" className="train-window" />
                  <rect x={x + coachWidthPx * .42} y="-8" width={Math.max(5, coachWidthPx * .16)} height="6" rx="1.5" className="train-window" />
                  <rect x={x + coachWidthPx * .66} y="-8" width={Math.max(5, coachWidthPx * .16)} height="6" rx="1.5" className="train-window" />
                  {!isFront && <rect x={x + coachWidthPx - 7} y="0" width="4" height="6" rx="1" className="train-door" />}
                  {isFront && <path d={`M ${coachWidthPx / 2 - 9} -12 L ${coachWidthPx / 2} -6 L ${coachWidthPx / 2} 5 L ${coachWidthPx / 2 - 9} 8 Z`} className="train-cab" />}
                  {!isRear && <line x1={x - COACH_GAP_PX / 2} y1="0" x2={x} y2="0" className="train-coupler" />}
                  <circle cx={x + Math.min(9, coachWidthPx * .24)} cy="10" r="3" />
                  <circle cx={x + coachWidthPx - Math.min(9, coachWidthPx * .24)} cy="10" r="3" />
                </g>
              })}
              <g transform={`translate(${labelX} ${labelY})`} className="train-label-anchor">
                <rect x="-72" y="-12" width="144" height="20" rx="8" className="train-label-bg" />
                <text x="0" y="-1" textAnchor="middle" dominantBaseline="middle" className="train-speed-label">{frame.train_id} · {Math.round(frame.speed_kmh)}</text>
              </g>
            </g>
          })}

          <text x={LEFT} y={HEIGHT - 16} className="endpoint-svg-label">DELHI SIDE</text>
          <text x={RIGHT} y={HEIGHT - 16} textAnchor="end" className="endpoint-svg-label">AGRA SIDE</text>
        </svg>
      </div>

      <div className="legend">
        <span>dynamic signals = block occupancy</span>
        <span>platforms follow their configured physical length</span>
        <span>⇄ = physical crossover</span>
        <span>reverse trains stay upright</span>
        <span className="crossing-legend road-open-legend">● road open = train stop before crossing</span>
        <span className="crossing-legend road-closed-legend">● road closed = rail protected</span>
        <span><i className="legend-swatch tsr-swatch" />TSR</span>
        <span><i className="legend-swatch maintenance-swatch" />Maintenance</span>
      </div>
    </section>
  )
}
