import { useEffect, useRef } from "react";

import { connectRunEvents, type RunEvent } from "./api";

const EVENT_TYPES = [
  "run_started",
  "graph_node_started",
  "graph_node_finished",
  "search_completed",
  "candidate_selected",
  "source_read",
  "source_rejected",
  "evidence_assessed",
  "generation_status",
  "run_completed",
  "run_failed",
] as const;

export function useRunEvents(
  runId: string | undefined,
  onEvent: (event: RunEvent) => void,
  onConnectionError?: () => void,
) {
  const onEventRef = useRef(onEvent);
  const onConnectionErrorRef = useRef(onConnectionError);
  onEventRef.current = onEvent;
  onConnectionErrorRef.current = onConnectionError;

  useEffect(() => {
    if (!runId) {
      return;
    }
    if (typeof EventSource === "undefined") {
      return;
    }
    const source = connectRunEvents(runId);
    const receive = (message: MessageEvent) => {
      try {
        onEventRef.current({
          eventId: Number(message.lastEventId),
          type: message.type,
          data: JSON.parse(message.data) as Record<string, unknown>,
        });
      } catch {
        onConnectionErrorRef.current?.();
      }
    };
    for (const eventType of EVENT_TYPES) {
      source.addEventListener(eventType, receive);
    }
    source.onerror = () => onConnectionErrorRef.current?.();
    return () => source.close();
  }, [runId]);
}
