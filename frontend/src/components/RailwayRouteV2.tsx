import type { SignalAspect, SimulatorConfigViz, TelemetryFrame, TrackBlockViz } from '../types'

interface Point {
  x: number
  y: number
}

interface Pose extends Point {
  angleDeg: number
}

const WIDTH = 1400
const HEIGHT = 360
const LEFT = 80
const RIGHT = WIDTH - 80
const BASE_Y = 185
const TRACK_SPACING = 18
const PATH_SAMPLES = 220

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

function curveDelta(block: TrackBlockViz): number {
  if (!block.curve_radius_m || !block.curve_direction) return 0
  const severity = Math.max(30, Math.min(72, 52000 / block.curve_radius_m))
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
    x:
      3 * u * u * (p1.x - p0.x) +
      6 * u * t * (p2.x - p1.x) +
      3 * t * t * (p3.x - p2.x),
    y:
      3 * u * u * (p1.y - p0.y) +
      6 * u * t * (p2.y - p1.y) +
      3 * t * t * (p3.y - p2.y),
  }
}

function centerPoseAt(config: SimulatorConfigViz, routePositionM: number): Pose {
  const blocks = geometry(config)
  const clamped = Math.max(0, Math.min(routePositionM, config.route.total_length_m))
  const item = blocks.find(({ block }) => clamped <= block.route_end_m) ?? blocks[blocks.length - 1]
  const span = Math.max(1, item.block.length_m)
  const t = Math.max(0, Math.min(1, (clamped - item.block.route_start_m) / span))

  const p0 = { x: item.x0, y: item.y0 }
  const p3 = { x: item.x1, y: item.y1 }
  let point: Point
  let tangent: Point

  if (item.block.curve_direction && item.block.curve_radius_m) {
    const dx = item.x1 - item.x0
    const p1 = { x: item.x0 + dx * 0.34, y: item.y0 }
    const p2 = { x: item.x0 + dx * 0.66, y: item.y1 }
    point = bezierPoint(p0, p1, p2, p3, t)
    tangent = bezierTangent(p0, p1, p2, p3, t)
  } else {
    point = {
      x: item.x0 + (item.x1 - item.x0) * t,
      y: item.y0 + (item.y1 - item.y0) * t,
    }
    tangent = { x: item.x1 - item.x0, y: item.y1 - item.y0 }
  }

  return {
    ...point,
    angleDeg: (Math.atan2(tangent.y, tangent.x) * 180) / Math.PI,
  }
}

function poseAt(config: SimulatorConfigViz, routePositionM: number, trackIndex: number): Pose {
  const center = centerPoseAt(config, routePositionM)
  const angleRad = (center.angleDeg * Math.PI) / 180
  const offset = (trackIndex - (config.route.track_ids.length - 1) / 2) * TRACK_SPACING
  const normalX = -Math.sin(angleRad)
  const normalY = Math.cos(angleRad)
  return {
    x: center.x + normalX * offset,
    y: center.y + normalY * offset,
    angleDeg: center.angleDeg,
  }
}

function pointAt(config: SimulatorConfigViz, routePositionM: number, trackIndex: number): Point {
  const { x, y } = poseAt(config, routePositionM, trackIndex)
  return { x, y }
}

function trackPath(config: SimulatorConfigViz, trackIndex: number): string {
  const points: Point[] = []
  for (let i = 0; i <= PATH_SAMPLES; i += 1) {
    const routeM = (i / PATH_SAMPLES) * config.route.total_length_m
    points.push(pointAt(config, routeM, trackIndex))
  }
  return points.map((p, index) => `${index === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ')
}

function trackIndex(config: SimulatorConfigViz, trackId: string): number {
  const index = config.route.track_ids.indexOf(trackId)
  return index >= 0 ? index : 0
}

function segmentPath(config: SimulatorConfigViz, startM: number, endM: number, trackIdx: number): string {
  const span = Math.max(1, endM - startM)
  const samples = Math.max(8, Math.ceil((span / config.route.total_length_m) * PATH_SAMPLES))
  const points: Point[] = []
  for (let i = 0; i <= samples; i += 1) {
    points.push(pointAt(config, startM + (i / samples) * span, trackIdx))
  }
  return points.map((p, index) => `${index === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ')
}

export default function RailwayRouteV2({ config, frame }: { config: SimulatorConfigViz; frame: TelemetryFrame }) {
  const trainTrackIndex = trackIndex(config, config.train.track_id)
  const trainPose = poseAt(config, frame.route_position_m, trainTrackIndex)
  const blocks = geometry(config)

  return (
    <section className="panel route-panel realistic-route-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Multi-track infrastructure view</span>
          <h2>{config.route.route_name}</h2>
        </div>
        <div className="route-meta">
          {config.route.track_ids.length} tracks · {(config.route.total_length_m / 1000).toFixed(1)} km
        </div>
      </div>

      <div className="realistic-route-scroll">
        <svg className="realistic-route" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Curved multi-track railway route">
          <g className="route-grid">
            {blocks.slice(1).map(({ block }) => {
              const p = pointAt(config, block.route_start_m, 0)
              return <line key={block.block_id} x1={p.x} y1={48} x2={p.x} y2={HEIGHT - 44} className="block-boundary" />
            })}
          </g>

          {config.route.track_ids.map((id, index) => (
            <g key={id}>
              <path d={trackPath(config, index)} className="track-ballast-path" />
              <path d={trackPath(config, index)} className="track-bed-path" />
              <path d={trackPath(config, index)} className="track-sleeper-path" />
              <path d={trackPath(config, index)} className="track-rail-path" />
              <text x={18} y={pointAt(config, 0, index).y + 4} className="track-name">{id}</text>
            </g>
          ))}

          {blocks.map(({ block }) => {
            const mid = pointAt(config, block.route_start_m + block.length_m / 2, 0)
            return (
              <g key={block.block_id} className="block-svg-label">
                <text x={mid.x} y={Math.max(25, mid.y - 58)} textAnchor="middle">{block.block_id}</text>
                <text x={mid.x} y={Math.max(38, mid.y - 43)} textAnchor="middle" className="block-speed-label">
                  {block.speed_limit_kmh} km/h
                </text>
                {block.curve_radius_m && (
                  <text x={mid.x} y={Math.max(51, mid.y - 28)} textAnchor="middle" className="curve-label">
                    {block.curve_direction === 'LEFT' ? '↶' : '↷'} R{block.curve_radius_m}m · {block.curve_speed_limit_kmh ?? block.speed_limit_kmh} km/h
                  </text>
                )}
              </g>
            )
          })}

          {config.environment.temporary_speed_restrictions.map((item) => (
            <g key={item.restriction_id}>
              <path d={segmentPath(config, item.route_start_m, item.route_end_m, trainTrackIndex)} className="restriction-line tsr-line" />
              {(() => {
                const mid = pointAt(config, (item.route_start_m + item.route_end_m) / 2, trainTrackIndex)
                return <text x={mid.x} y={mid.y + 29} textAnchor="middle" className="restriction-svg-label">TSR {item.speed_limit_kmh}</text>
              })()}
            </g>
          ))}

          {config.environment.maintenance_restrictions.map((item) => (
            <g key={item.restriction_id}>
              <path d={segmentPath(config, item.route_start_m, item.route_end_m, trainTrackIndex)} className="restriction-line maintenance-line" />
              {(() => {
                const mid = pointAt(config, (item.route_start_m + item.route_end_m) / 2, trainTrackIndex)
                return <text x={mid.x} y={mid.y + 42} textAnchor="middle" className="restriction-svg-label">MNT {item.speed_limit_kmh}</text>
              })()}
            </g>
          ))}

          {config.signals.map((signal) => {
            const index = trackIndex(config, signal.track_id)
            const p = pointAt(config, signal.route_position_m, index)
            const aspect = signalAspect(config, signal.signal_id, frame.sim_time_s)
            return (
              <g key={signal.signal_id} transform={`translate(${p.x} ${p.y - 2})`} className="svg-signal">
                <line x1="0" y1="0" x2="0" y2="-28" />
                <circle cx="0" cy="-34" r="7" className={`svg-signal-light ${aspect.toLowerCase()}`} />
                <text x="0" y="-47" textAnchor="middle">{signal.signal_id}</text>
              </g>
            )
          })}

          {config.environment.crossings.map((crossing) => {
            const state = activeAt(crossing.timeline, frame.sim_time_s)?.state ?? 'OPEN_FOR_TRAIN'
            const p0 = pointAt(config, crossing.route_position_m, 0)
            const p1 = pointAt(config, crossing.route_position_m, config.route.track_ids.length - 1)
            return (
              <g key={crossing.crossing_id} className={state === 'CLOSED_FOR_TRAIN' ? 'svg-crossing closed' : 'svg-crossing'}>
                <line x1={p0.x} y1={p0.y - 18} x2={p1.x} y2={p1.y + 18} />
                <text x={p0.x + 6} y={Math.max(p0.y, p1.y) + 35}>× {crossing.crossing_id}</text>
              </g>
            )
          })}

          {config.route.track_ids.map((_, index) => {
            const source = pointAt(config, config.journey.source_route_m, index)
            const dest = pointAt(config, config.journey.destination_route_m, index)
            return (
              <g key={`ends-${index}`} className="route-endpoints">
                <circle cx={source.x} cy={source.y} r="4" />
                <circle cx={dest.x} cy={dest.y} r="4" />
              </g>
            )
          })}

          <g transform={`translate(${trainPose.x} ${trainPose.y})`} className="svg-train-position">
            <g transform={`rotate(${trainPose.angleDeg})`} className="svg-train-body">
              <rect x="-22" y="-12" width="42" height="20" rx="5" />
              <path d="M 12 -12 L 22 -6 L 22 5 L 12 8 Z" />
              <rect x="-12" y="-8" width="8" height="6" rx="1" className="train-window" />
              <rect x="-1" y="-8" width="8" height="6" rx="1" className="train-window" />
              <circle cx="-11" cy="10" r="3" />
              <circle cx="11" cy="10" r="3" />
            </g>
            <text x="0" y="-27" textAnchor="middle" className="train-speed-label">{frame.speed_kmh.toFixed(1)} km/h</text>
          </g>

          <text x={LEFT} y={HEIGHT - 16} className="endpoint-svg-label">SOURCE</text>
          <text x={RIGHT} y={HEIGHT - 16} textAnchor="end" className="endpoint-svg-label">DESTINATION</text>
        </svg>
      </div>

      <div className="legend">
        <span><i className="legend-swatch tsr-swatch" />TSR</span>
        <span><i className="legend-swatch maintenance-swatch" />Maintenance</span>
        <span>parallel rails follow the same centreline geometry</span>
        <span>train rotates to the live track tangent</span>
        <span>train on {config.train.track_id}</span>
      </div>
    </section>
  )
}
