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
  evidence: {
    candidates: SourceSnapshot[];
    read_sources: SourceSnapshot[];
    rejected_sources: SourceSnapshot[];
  };
};

export async function getRun(runId: string): Promise<RunSnapshot> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) {
    throw new Error(`Unable to load run (${response.status})`);
  }
  return (await response.json()) as RunSnapshot;
}
