import type { ManualInjectionRequest, SessionCreateResponse, SocketMessage } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const WS_BASE = API_BASE.replace(/^http/, 'ws')

export async function createSession(
  scenarioName = 'delhi_agra_corridor.yaml',
): Promise<SessionCreateResponse> {
  const response = await fetch(`${API_BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_name: scenarioName }),
  })
  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`)
  }
  return response.json() as Promise<SessionCreateResponse>
}

export function connectSimulation(
  sessionId: string,
  onMessage: (message: SocketMessage) => void,
  onStatus: (status: 'connecting' | 'open' | 'closed' | 'error') => void,
): WebSocket {
  onStatus('connecting')
  const socket = new WebSocket(`${WS_BASE}/ws/${sessionId}`)
  socket.onopen = () => onStatus('open')
  socket.onclose = () => onStatus('closed')
  socket.onerror = () => onStatus('error')
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data) as SocketMessage)
    } catch {
      onMessage({ type: 'error', message: 'Received malformed server message' })
    }
  }
  return socket
}

export function sendCommand(
  socket: WebSocket | null,
  command: 'play' | 'pause' | 'reset' | 'step' | 'set_speed',
  speed?: number,
): void {
  if (!socket || socket.readyState !== WebSocket.OPEN) return
  socket.send(JSON.stringify(speed === undefined ? { command } : { command, speed }))
}

export function sendInjection(socket: WebSocket | null, request: ManualInjectionRequest): void {
  if (!socket || socket.readyState !== WebSocket.OPEN) return
  socket.send(JSON.stringify(request))
}
