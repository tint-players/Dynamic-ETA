import type { SignalAspect, SimulatorConfigViz, TelemetryFrame, TrackBlockViz } from '../types'

interface Point {
  x: number
  y: number
}

const WIDTH = 1400
const HEIGHT = 360
const LEFT = 80
const RIGHT = WIDTH - 80
const BASE_Y = 185
const TRACK_SPACING = 18

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
  const blocks = config.route.blocks.map((block) => {
    const x0 = LEFT + (block.route_start_m / total) * (RIGHT - LEFT)
    const x1 = LEFT + (block.route_end_m / total) * (RIGHT - LEFT)
    const y0 = y
    const y1 = y + curveDelta(block)
    y = y1
    return { block, x0, x1, y0, y1 }
  })
  return blocks
}

function pointAt(config: SimulatorConfigViz, routePositionM: number, trackIndex: number): Point {
  const blocks = geometry(config)
  const clamped = Math.max(0, Math.min(routePositionM, config.route.total_length_m))
  const item = blocks.find(({ block }) => clamped <= block.route_end_m) ?? blocks[blocks.length - 1]
  const span = Math.max(1, item.block.length_m)
  const t = Math.max(0, Math.min(1, (clamped - item.block.route_start_m) / span))
  const smooth = t * t * (3 - 2 * t)
  const x = item.x0 + (item.x1 - item.x0) * t
  const y = item.y0 + (item.y1 - item.y0) * smooth
  const centeredOffset = (trackIndex - (config.route.track_ids.length - 1) / 2) * TRACK_SPACING
  return { x, y: y + centeredOffset }
}

function trackPath(config: SimulatorConfigViz, trackIndex: number): string {
  const blocks = geometry(config)
  const centeredOffset = (trackIndex - (config.route.track_ids.length - 1) / 2) * TRACK_SPACING
  if (!blocks.length) return ''
  let d = `M ${blocks[0].x0} ${blocks[0].y0 + centeredOffset}`
  for (const item of blocks) {
    if (item.block.curve_direction && item.block.curve_radius_m) {
      const dx = item.x1 - item.x0
      d += ` C ${item.x0 + dx * 0.34} ${item.y0 + centeredOffset}, ${item.x0 + dx * 0.66} ${item.y1 + centeredOffset}, ${item.x1} ${item.y1 + centeredOffset}`
    } else {
      d += ` L ${item.x1} ${item.y1 + centeredOffset}`
    }
  }
  return d
}

function trackIndex(config: SimulatorConfigViz, trackId: string): number {
  const index = config.route.track_ids.indexOf(trackId)
  return index >= 0 ? index : 0
}

export default function RailwayRouteV2({ config, frame }: { config: SimulatorConfigViz; frame: TelemetryFrame }) {
  const trainTrackIndex = trackIndex(config, config.train.track_id)
  const trainPoint = pointAt(config, frame.route_position_m, trainTrackIndex)
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
              <path d={trackPath(config, index)} className="track-bed-path" />
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
                    ↪ R{block.curve_radius_m}m · {block.curve_speed_limit_kmh ?? block.speed_limit_kmh} km/h
                  </text>
                )}
              </g>
            )
          })}

          {config.environment.temporary_speed_restrictions.map((item) => {
            const a = pointAt(config, item.route_start_m, trainTrackIndex)
            const b = pointAt(config, item.route_end_m, trainTrackIndex)
            return (
              <g key={item.restriction_id}>
                <line x1={a.x} y1={a.y + 10} x2={b.x} y2={b.y + 10} className="restriction-line tsr-line" />
                <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 + 27} textAnchor="middle" className="restriction-svg-label">TSR {item.speed_limit_kmh}</text>
              </g>
            )
          })}

          {config.environment.maintenance_restrictions.map((item) => {
            const a = pointAt(config, item.route_start_m, trainTrackIndex)
            const b = pointAt(config, item.route_end_m, trainTrackIndex)
            return (
              <g key={item.restriction_id}>
                <line x1={a.x} y1={a.y + 17} x2={b.x} y2={b.y + 17} className="restriction-line maintenance-line" />
                <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 + 39} textAnchor="middle" className="restriction-svg-label">MNT {item.speed_limit_kmh}</text>
              </g>
            )
          })}

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

          <g transform={`translate(${trainPoint.x} ${trainPoint.y})`} className="svg-train">
            <rect x="-19" y="-13" width="38" height="22" rx="6" />
            <path d="M 12 -13 L 20 -6 L 20 5 L 12 9 Z" />
            <circle cx="-10" cy="11" r="3" />
            <circle cx="10" cy="11" r="3" />
            <text x="0" y="-23" textAnchor="middle">{frame.speed_kmh.toFixed(1)} km/h</text>
          </g>

          <text x={LEFT} y={HEIGHT - 16} className="endpoint-svg-label">SOURCE</text>
          <text x={RIGHT} y={HEIGHT - 16} textAnchor="end" className="endpoint-svg-label">DESTINATION</text>
        </svg>
      </div>

      <div className="legend">
        <span><i className="legend-swatch tsr-swatch" />TSR</span>
        <span><i className="legend-swatch maintenance-swatch" />Maintenance</span>
        <span>parallel rails = separate tracks</span>
        <span>curved geometry follows block curve metadata</span>
        <span>train on {config.train.track_id}</span>
      </div>
    </section>
  )
}
