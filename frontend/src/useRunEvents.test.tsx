import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useRunEvents } from "./useRunEvents";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  private listeners = new Map<string, (event: MessageEvent) => void>();
  closed = false;

  constructor(_url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    this.listeners.set(type, listener);
  }

  close() {
    this.closed = true;
  }

  emit(type: string, data: object, lastEventId = "1") {
    this.listeners.get(type)?.(
      new MessageEvent(type, { data: JSON.stringify(data), lastEventId }),
    );
  }
}

function Probe({ onEvent }: { onEvent: (eventType: string) => void }) {
  useRunEvents("run-1", (event) => onEvent(event.type));
  return <p>listening</p>;
}

describe("useRunEvents", () => {
  afterEach(() => {
    FakeEventSource.instances = [];
    vi.unstubAllGlobals();
  });

  it("receives a typed event and closes the connection on unmount", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn();
    const view = render(<Probe onEvent={onEvent} />);

    FakeEventSource.instances[0].emit("source_read", { source_id: "source-1" });

    expect(onEvent).toHaveBeenCalledWith("source_read");
    expect(screen.getByText("listening")).toBeInTheDocument();
    view.unmount();
    expect(FakeEventSource.instances[0].closed).toBe(true);
  });
});
