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
  question?: string;
  evidence_status?: string | null;
  evidence_policy?: string | null;
  answer?: string | null;
  interrupt_prompt?: string | null;
  evidence: {
    candidates: SourceSnapshot[];
    read_sources: SourceSnapshot[];
    rejected_sources: SourceSnapshot[];
  };
};

export type SessionSnapshot = {
  session_id: string;
  runs: RunSnapshot[];
};

export type SessionListItem = {
  session_id: string;
  display_name: string;
  updated_at?: string;
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

export async function listSessions(): Promise<SessionListItem[]> {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    throw new Error(`Unable to load sessions (${response.status})`);
  }
  return (await response.json()) as SessionListItem[];
}

export async function renameSession(
  sessionId: string,
  displayName: string,
): Promise<SessionListItem> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!response.ok) {
    throw new Error(`Unable to rename session (${response.status})`);
  }
  return (await response.json()) as SessionListItem;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Unable to delete session (${response.status})`);
  }
}

export async function getSession(sessionId: string): Promise<SessionSnapshot> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    throw new Error(`Unable to load session (${response.status})`);
  }
  return (await response.json()) as SessionSnapshot;
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

export async function resumeTurn(
  sessionId: string,
  answer: string,
): Promise<RunSnapshot> {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/resume`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify({ answer }),
    },
  );
  if (!response.ok) {
    throw new Error(`Unable to resume research (${response.status})`);
  }
  return (await response.json()) as RunSnapshot;
}
