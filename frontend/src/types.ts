export type SignalAspect = 'GREEN' | 'YELLOW' | 'RED'
export type WeatherCondition = 'CLEAR' | 'RAIN' | 'HEAVY_RAIN' | 'FOG' | 'HEAVY_FOG'
export type CrossingState = 'OPEN_FOR_TRAIN' | 'CLOSED_FOR_TRAIN'
export type TrainDirection = 'FORWARD' | 'REVERSE'

export interface TrackBlockViz {
  block_id: string
  length_m: number
  speed_limit_kmh: number
  gradient_percent: number
  curve_radius_m: number | null
  curve_speed_limit_kmh: number | null
  curve_direction: 'LEFT' | 'RIGHT' | null
  route_start_m: number
  route_end_m: number
}

export interface SignalViz {
  signal_id: string
  protected_block_id: string
  track_id: string
  direction: TrainDirection
  route_position_m: number
}

export interface RestrictionViz {
  restriction_id: string
  block_id: string
  start_position_m: number
  end_position_m: number
  speed_limit_kmh: number
  start_time_s: number
  end_time_s: number | null
  route_start_m: number
  route_end_m: number
}

export interface CrossingViz {
  crossing_id: string
  block_id: string
  position_in_block_m: number
  route_position_m: number
  timeline: Array<{ start_time_s: number; state: CrossingState }>
}

export interface CrossoverViz {
  crossover_id: string
  block_id: string
  start_position_m: number
  end_position_m: number
  from_track_id: string
  to_track_id: string
  route_start_m: number
  route_end_m: number
  route_mid_m: number
}

export interface SignalSchedule { signal_id: string; timeline: Array<{ start_time_s: number; aspect: SignalAspect }> }
export interface WeatherSchedule { block_id: string; timeline: Array<{ start_time_s: number; condition: WeatherCondition; visibility_m: number }> }

export interface TrainConfigViz {
  train_id: string
  train_name: string
  track_id: string
  max_speed_kmh: number
  length_m: number
  accel_ms2: number
  service_decel_ms2: number
  emergency_decel_ms2: number
}

export interface StationPlatformViz {
  platform_id: string
  track_id: string
  block_id: string
  position_in_block_m: number
  length_m: number
  route_position_m: number
}

export interface StationViz { station_id: string; station_name: string; platforms: StationPlatformViz[] }

export interface TrainRunViz {
  train: TrainConfigViz
  journey: { source: { block_id: string; position_in_block_m: number }; destination: { block_id: string; position_in_block_m: number } }
  departure_time_s: number
  station_stops: Array<{ station_id: string; dwell_time_s: number }>
  track_changes: Array<{ crossover_id: string; reverse_after_change: boolean }>
  source_route_m: number
  destination_route_m: number
}

export interface SimulatorConfigViz {
  route: { route_id: string; route_name: string; track_ids: string[]; total_length_m: number; blocks: TrackBlockViz[] }
  signals: SignalViz[]
  dynamic_signalling: boolean
  stations: StationViz[]
  crossovers: CrossoverViz[]
  trains: TrainRunViz[]
  train: TrainConfigViz
  journey: {
    source: { block_id: string; position_in_block_m: number }
    destination: { block_id: string; position_in_block_m: number }
    source_route_m: number
    destination_route_m: number
  }
  environment: {
    weather: WeatherSchedule[]
    signal_states: SignalSchedule[]
    temporary_speed_restrictions: RestrictionViz[]
    maintenance_restrictions: RestrictionViz[]
    crossings: CrossingViz[]
  }
  simulation: {
    tick_seconds: number
    max_simulation_time_s: number
    random_seed: number | null
    braking_safety_margin_m: number
    speed_tolerance_kmh: number
    train_separation_m: number
  }
}

export interface TelemetryFrame {
  scenario_id: string
  train_id: string
  tick: number
  sim_time_s: number
  track_id: string
  direction: TrainDirection
  active: boolean
  completed: boolean
  current_station_id: string | null
  current_block_id: string
  position_in_block_m: number
  route_position_m: number
  speed_kmh: number
  acceleration_ms2: number
  control_action: string
  control_reason: string
  current_block_speed_limit_kmh: number
  current_gradient_percent: number
  current_curve_speed_limit_kmh: number | null
  effective_speed_ceiling_kmh: number
  weather: WeatherCondition
  visibility_m: number
  distance_to_destination_m: number
  route_progress: number
  next_signal_aspect: SignalAspect | null
  distance_to_next_signal_m: number | null
  second_signal_aspect: SignalAspect | null
  distance_to_second_signal_m: number | null
  next_speed_limit_kmh: number | null
  distance_to_next_speed_change_m: number | null
  next_curve_limit_kmh: number | null
  distance_to_next_curve_m: number | null
  next_tsr_limit_kmh: number | null
  distance_to_next_tsr_m: number | null
  next_crossing_state: CrossingState | null
  distance_to_next_crossing_m: number | null
  actual_remaining_time_s: number | null
  actual_arrival_simulation_s: number | null
  total_journey_time_s: number | null
}

export interface ActiveWeatherConstraint {
  block_id: string
  condition: WeatherCondition
  visibility_m: number
  start_time_s: number
  end_time_s: number | null
}

export interface ActiveRestrictionConstraint {
  restriction_id: string
  block_id: string
  start_position_m: number
  end_position_m: number
  speed_limit_kmh: number
  start_time_s: number
  end_time_s: number | null
}

export interface ActiveSignalConstraint {
  signal_id: string
  aspect: SignalAspect
  start_time_s: number | null
  end_time_s: number | null
}

export interface ConstraintState {
  sim_time_s: number
  weather: ActiveWeatherConstraint[]
  tsr: ActiveRestrictionConstraint[]
  maintenance: ActiveRestrictionConstraint[]
  signals: ActiveSignalConstraint[]
}

export const EMPTY_CONSTRAINT_STATE: ConstraintState = { sim_time_s: 0, weather: [], tsr: [], maintenance: [], signals: [] }

export type ManualInjectionRequest =
  | {
      command: 'inject_weather'
      block_id: string
      condition: WeatherCondition
      visibility_m: number
      duration_s: number | null
    }
  | {
      command: 'inject_signal'
      signal_id: string
      aspect: SignalAspect
      duration_s: number | null
    }
  | {
      command: 'reset_signal'
      signal_id: string
    }
  | {
      command: 'inject_tsr' | 'inject_maintenance'
      block_id: string
      start_position_m: number
      end_position_m: number
      speed_limit_kmh: number
      duration_s: number | null
    }

export interface SessionCreateResponse {
  session_id: string
  scenario_id: string
  scenario_name: string
  config: SimulatorConfigViz
  initial_frame: TelemetryFrame
  initial_frames: TelemetryFrame[]
  signal_states: Record<string, SignalAspect>
  crossing_states: Record<string, CrossingState>
  constraint_state: ConstraintState
}

export interface PlaybackState { playing: boolean; playback_speed: number; complete: boolean }
export interface ExportPaths { csv: string; parquet: string; block_csv?: string; block_parquet?: string }

export type SocketMessage =
  | { type: 'telemetry_batch'; frames: TelemetryFrame[]; signal_states: Record<string, SignalAspect>; crossing_states: Record<string, CrossingState> }
  | ({ type: 'playback_state' } & PlaybackState)
  | { type: 'config_update'; config: SimulatorConfigViz }
  | { type: 'constraint_state'; state: ConstraintState }
  | { type: 'constraints_reset'; sim_time_s: number }
  | { type: 'injection_applied'; message: string; sim_time_s: number }
  | { type: 'export_complete'; paths: ExportPaths }
  | { type: 'error'; message: string }
