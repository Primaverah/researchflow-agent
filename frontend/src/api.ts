export type SourceSnapshot = {
  source_id: string;
  title: string;
  url: string;
  kind: string;
  read: boolean;
  reason: string | null;
};

export type RunSnapshot = {
  run_id: string;
  session_id: string;
  status: string;
  answer?: string | null;
  evidence: {
    candidates: SourceSnapshot[];
    read_sources: SourceSnapshot[];
    rejected_sources: SourceSnapshot[];
  };
};

export type RunEvent = {
  eventId: number;
  type: string;
  data: Record<string, unknown>;
};

export function connectRunEvents(runId: string): EventSource {
  return new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
}

export async function getRun(runId: string): Promise<RunSnapshot> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) {
    throw new Error(`Unable to load run (${response.status})`);
  }
  return (await response.json()) as RunSnapshot;
}

export async function listSessions(): Promise<string[]> {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    throw new Error(`Unable to load sessions (${response.status})`);
  }
  return (await response.json()) as string[];
}

export async function startTurn(
  sessionId: string,
  message: string,
): Promise<RunSnapshot> {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/turns`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify({ message }),
    },
  );
  if (!response.ok) {
    throw new Error(`Unable to start research (${response.status})`);
  }
  return (await response.json()) as RunSnapshot;
}
