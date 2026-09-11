import type { SignalAspect, SimulatorConfigViz, TelemetryFrame, TrackBlockViz } from '../types'

interface Point { x: number; y: number }
interface Pose extends Point { angleDeg: number }

const WIDTH = 1400
const HEIGHT = 430
const LEFT = 80
const RIGHT = WIDTH - 80
const BASE_Y = 220
const TRACK_SPACING = 24
const PATH_SAMPLES = 260

function activeAt<T extends { start_time_s: number }>(timeline: T[], simTime: number): T | undefined {
  let active: T | undefined
  for (const entry of timeline) { if (entry.start_time_s <= simTime) active = entry; else break }
  return active
}

function curveDelta(block: TrackBlockViz): number {
  if (!block.curve_radius_m || !block.curve_direction) return 0
  const severity = Math.max(28, Math.min(70, 52000 / block.curve_radius_m))
  return block.curve_direction === 'RIGHT' ? severity : -severity
}

function geometry(config: SimulatorConfigViz) {
  const total = config.route.total_length_m; let y = BASE_Y
  return config.route.blocks.map((block) => {
    const x0 = LEFT + (block.route_start_m / total) * (RIGHT - LEFT)
    const x1 = LEFT + (block.route_end_m / total) * (RIGHT - LEFT)
    const y0 = y; const y1 = y + curveDelta(block); y = y1
    return { block, x0, x1, y0, y1 }
  })
}

function bezierPoint(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u = 1-t, uu=u*u, tt=t*t
  return { x: uu*u*p0.x + 3*uu*t*p1.x + 3*u*tt*p2.x + tt*t*p3.x, y: uu*u*p0.y + 3*uu*t*p1.y + 3*u*tt*p2.y + tt*t*p3.y }
}
function bezierTangent(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u=1-t
  return { x: 3*u*u*(p1.x-p0.x)+6*u*t*(p2.x-p1.x)+3*t*t*(p3.x-p2.x), y: 3*u*u*(p1.y-p0.y)+6*u*t*(p2.y-p1.y)+3*t*t*(p3.y-p2.y) }
}
function centerPoseAt(config: SimulatorConfigViz, routePositionM: number): Pose {
  const blocks=geometry(config); const clamped=Math.max(0,Math.min(routePositionM,config.route.total_length_m))
  const item=blocks.find(({block})=>clamped<=block.route_end_m)??blocks[blocks.length-1]
  const t=Math.max(0,Math.min(1,(clamped-item.block.route_start_m)/Math.max(1,item.block.length_m)))
  const p0={x:item.x0,y:item.y0}, p3={x:item.x1,y:item.y1}; let point:Point,tangent:Point
  if(item.block.curve_direction&&item.block.curve_radius_m){const dx=item.x1-item.x0;const p1={x:item.x0+dx*.34,y:item.y0},p2={x:item.x0+dx*.66,y:item.y1};point=bezierPoint(p0,p1,p2,p3,t);tangent=bezierTangent(p0,p1,p2,p3,t)}
  else {point={x:item.x0+(item.x1-item.x0)*t,y:item.y0+(item.y1-item.y0)*t};tangent={x:item.x1-item.x0,y:item.y1-item.y0}}
  return {...point,angleDeg:Math.atan2(tangent.y,tangent.x)*180/Math.PI}
}
function poseAt(config: SimulatorConfigViz, routePositionM:number, trackIndex:number):Pose{
  const center=centerPoseAt(config,routePositionM);const angle=center.angleDeg*Math.PI/180;const offset=(trackIndex-(config.route.track_ids.length-1)/2)*TRACK_SPACING
  return {x:center.x-Math.sin(angle)*offset,y:center.y+Math.cos(angle)*offset,angleDeg:center.angleDeg}
}
function trackIndex(config:SimulatorConfigViz,trackId:string){const i=config.route.track_ids.indexOf(trackId);return i>=0?i:0}
function pointAt(config:SimulatorConfigViz,routePositionM:number,trackIdx:number):Point{const{x,y}=poseAt(config,routePositionM,trackIdx);return{x,y}}
function pathBetween(config:SimulatorConfigViz,startM:number,endM:number,trackIdx:number,samples=PATH_SAMPLES){const span=Math.max(1,endM-startM);const count=Math.max(8,Math.ceil((span/config.route.total_length_m)*samples));const points:Point[]=[];for(let i=0;i<=count;i+=1)points.push(pointAt(config,startM+span*i/count,trackIdx));return points.map((p,i)=>`${i===0?'M':'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ')}
function fullTrackPath(config:SimulatorConfigViz,trackIdx:number){return pathBetween(config,0,config.route.total_length_m,trackIdx)}
function signalFallback(config:SimulatorConfigViz,signalId:string,simTime:number):SignalAspect{const schedule=config.environment.signal_states.find((item)=>item.signal_id===signalId);return activeAt(schedule?.timeline??[],simTime)?.aspect??'GREEN'}

function displayPose(config: SimulatorConfigViz, frame: TelemetryFrame): Pose {
  const run = config.trains.find((item) => item.train.train_id === frame.train_id)
  const plan = run?.track_changes[0]
  const crossover = plan ? config.crossovers.find((item) => item.crossover_id === plan.crossover_id) : undefined
  const hasCompletedTurnaround = Boolean(plan?.reverse_after_change && frame.direction === 'REVERSE' && frame.track_id === crossover?.to_track_id)
  if (crossover && !hasCompletedTurnaround && frame.route_position_m >= crossover.route_start_m && frame.route_position_m <= crossover.route_end_m) {
    const t = (frame.route_position_m - crossover.route_start_m) / Math.max(1, crossover.route_end_m - crossover.route_start_m)
    const from = poseAt(config, frame.route_position_m, trackIndex(config, crossover.from_track_id))
    const to = poseAt(config, frame.route_position_m, trackIndex(config, crossover.to_track_id))
    return { x: from.x + (to.x-from.x)*t, y: from.y + (to.y-from.y)*t, angleDeg: from.angleDeg }
  }
  return poseAt(config, frame.route_position_m, trackIndex(config, frame.track_id))
}

export default function RailwayRouteV2({config,frames,signalStates,selectedTrainId,onSelectTrain}:{config:SimulatorConfigViz;frames:TelemetryFrame[];signalStates:Record<string,SignalAspect>;selectedTrainId:string;onSelectTrain:(trainId:string)=>void}){
  const simTime=frames[0]?.sim_time_s??0;const blocks=geometry(config)
  return <section className="panel route-panel realistic-route-panel"><div className="panel-heading"><div><span className="eyebrow">Live railway network</span><h2>{config.route.route_name}</h2></div><div className="route-meta">{config.route.track_ids.length} tracks · {frames.length} trains · {config.stations.length} stations</div></div><div className="realistic-route-scroll"><svg className="realistic-route" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Multi-train curved railway network">
    {blocks.slice(1).map(({block})=>{const p=pointAt(config,block.route_start_m,0);return <line key={block.block_id} x1={p.x} y1={50} x2={p.x} y2={HEIGHT-48} className="block-boundary"/>})}
    {config.route.track_ids.map((id,index)=><g key={id}><path d={fullTrackPath(config,index)} className="track-ballast-path"/><path d={fullTrackPath(config,index)} className="track-bed-path"/><path d={fullTrackPath(config,index)} className="track-sleeper-path"/><path d={fullTrackPath(config,index)} className="track-rail-path"/><text x={18} y={pointAt(config,0,index).y+4} className="track-name">{id}</text></g>)}

    {config.crossovers.map((xover)=>{const fromIdx=trackIndex(config,xover.from_track_id),toIdx=trackIndex(config,xover.to_track_id);const a=pointAt(config,xover.route_start_m,fromIdx),b=pointAt(config,xover.route_end_m,toIdx),c=pointAt(config,xover.route_start_m,toIdx),d=pointAt(config,xover.route_end_m,fromIdx);return <g key={xover.crossover_id} className="svg-crossover"><path d={`M ${a.x} ${a.y} L ${b.x} ${b.y}`}/><path d={`M ${c.x} ${c.y} L ${d.x} ${d.y}`}/><text x={(a.x+b.x)/2} y={(a.y+b.y)/2-12} textAnchor="middle">⇄ {xover.crossover_id}</text></g>})}

    {config.stations.map((station)=><g key={station.station_id} className="svg-station">{station.platforms.map((platform)=>{const idx=trackIndex(config,platform.track_id);const p=poseAt(config,platform.route_position_m,idx);const side=idx===0?-1:1;return <g key={platform.platform_id} transform={`translate(${p.x} ${p.y}) rotate(${p.angleDeg})`}><rect x="-23" y={side<0?-34:22} width="46" height="12" rx="3"/></g>})}{(()=>{const p=poseAt(config,station.platforms[0].route_position_m,0);return <g transform={`translate(${p.x} ${p.y}) rotate(${p.angleDeg})`}><text x="0" y="-49" textAnchor="middle">▰ {station.station_name}</text></g>})()}</g>)}

    {blocks.map(({block})=>{const p=pointAt(config,block.route_start_m+block.length_m/2,0);return <g key={block.block_id} className="block-svg-label"><text x={p.x} y={Math.max(22,p.y-76)} textAnchor="middle">{block.block_id}</text>{block.curve_radius_m&&<text x={p.x} y={Math.max(37,p.y-61)} textAnchor="middle" className="curve-label">{block.curve_direction==='LEFT'?'↶':'↷'} R{block.curve_radius_m}m</text>}</g>})}
    {config.environment.temporary_speed_restrictions.map((item)=><path key={item.restriction_id} d={pathBetween(config,item.route_start_m,item.route_end_m,0)} className="restriction-line tsr-line"/>)}
    {config.environment.maintenance_restrictions.map((item)=><path key={item.restriction_id} d={pathBetween(config,item.route_start_m,item.route_end_m,0)} className="restriction-line maintenance-line"/>)}

    {config.signals.map((signal)=>{const idx=trackIndex(config,signal.track_id);const p=poseAt(config,signal.route_position_m,idx);const aspect=signalStates[signal.signal_id]??signalFallback(config,signal.signal_id,simTime);const side=signal.direction==='FORWARD'?-1:1;return <g key={signal.signal_id} transform={`translate(${p.x} ${p.y}) rotate(${p.angleDeg})`} className="svg-signal"><line x1="0" y1="0" x2="0" y2={side*28}/><circle cx="0" cy={side*34} r="7" className={`svg-signal-light ${aspect.toLowerCase()}`}/><text x="0" y={side*47} textAnchor="middle" transform={side>0?'rotate(180 0 47)':undefined}>{signal.signal_id}</text></g>})}

    {config.environment.crossings.map((crossing)=>{const state=activeAt(crossing.timeline,simTime)?.state??'OPEN_FOR_TRAIN';const center=centerPoseAt(config,crossing.route_position_m);return <g key={crossing.crossing_id} transform={`translate(${center.x} ${center.y}) rotate(${center.angleDeg})`} className={state==='CLOSED_FOR_TRAIN'?'svg-crossing closed':'svg-crossing'}><line className="crossing-road-bed" x1="0" y1="-54" x2="0" y2="54"/><line className="crossing-road-mark" x1="0" y1="-54" x2="0" y2="54"/><line className="crossing-gate" x1="-18" y1="-38" x2="18" y2="-38"/><line className="crossing-gate" x1="-18" y1="38" x2="18" y2="38"/><text x="12" y="-58">{crossing.crossing_id}</text></g>})}

    {frames.map((frame,index)=>{const pose=displayPose(config,frame);const reverse=frame.direction==='REVERSE';const selected=frame.train_id===selectedTrainId;return <g key={frame.train_id} transform={`translate(${pose.x} ${pose.y})`} className={`svg-train-position train-${index%4} ${frame.active?'active':'waiting'} ${frame.completed?'completed':''} ${selected?'selected':''}`} onClick={()=>onSelectTrain(frame.train_id)}><g transform={`rotate(${pose.angleDeg}) ${reverse?'scale(-1 1)':''}`} className="svg-train-body"><rect x="-22" y="-12" width="42" height="20" rx="5"/><path d="M 12 -12 L 22 -6 L 22 5 L 12 8 Z"/><rect x="-12" y="-8" width="8" height="6" rx="1" className="train-window"/><rect x="-1" y="-8" width="8" height="6" rx="1" className="train-window"/><circle cx="-11" cy="10" r="3"/><circle cx="11" cy="10" r="3"/></g><text x="0" y="-28" textAnchor="middle" className="train-speed-label">{frame.train_id} · {frame.speed_kmh.toFixed(0)}</text></g>})}
    <text x={LEFT} y={HEIGHT-16} className="endpoint-svg-label">DELHI SIDE</text><text x={RIGHT} y={HEIGHT-16} textAnchor="end" className="endpoint-svg-label">AGRA SIDE</text>
  </svg></div><div className="legend"><span>dynamic signals = block occupancy</span><span>platforms and labels follow track slope</span><span>⇄ = physical crossover</span><span>reverse trains stay upright</span><span><i className="legend-swatch tsr-swatch"/>TSR</span><span><i className="legend-swatch maintenance-swatch"/>Maintenance</span></div></section>
}
