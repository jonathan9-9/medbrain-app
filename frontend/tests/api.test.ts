import { describe, expect, it } from "vitest";
import { parseSseBuffer } from "../lib/api";

describe("parseSseBuffer", () => {
  it("parses a complete SSE event", () => {
    const result = parseSseBuffer('event: status\ndata: "answered"\n\n');

    expect(result.events).toEqual([
      {
        type: "status",
        data: "answered",
      },
    ]);

    expect(result.remaining).toBe("");
  });

  it("handles events split across chunk boundaries", () => {
    const first = parseSseBuffer('event: token\ndata: "Hello');

    expect(first.events).toEqual([]);
    expect(first.remaining).toBe('event: token\ndata: "Hello');

    const second = parseSseBuffer(`${first.remaining} world"\n\n`);

    expect(second.events).toEqual([
      {
        type: "token",
        data: "Hello world",
      },
    ]);

    expect(second.remaining).toBe("");
  });

  it("parses multiple events from one buffer", () => {
    const result = parseSseBuffer(
      'event: status\ndata: "answered"\n\n' + 'event: token\ndata: "Hello"\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "status",
        data: "answered",
      },
      {
        type: "token",
        data: "Hello",
      },
    ]);
  });

  it("ignores malformed JSON events", () => {
    const result = parseSseBuffer("event: token\ndata: not-valid-json\n\n");

    expect(result.events).toEqual([]);
    expect(result.remaining).toBe("");
  });

  it("ignores events without an event type", () => {
    const result = parseSseBuffer('data: "hello"\n\n');

    expect(result.events).toEqual([]);
  });

  it("parses array payloads", () => {
    const result = parseSseBuffer(
      'event: retrieval\ndata: ["SOP-PS-002","SOP-CO-014"]\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "retrieval",
        data: ["SOP-PS-002", "SOP-CO-014"],
      },
    ]);
  });

  it("parses citation array payloads", () => {
    const result = parseSseBuffer(
      'event: citations\ndata: [{"tag":"S1","doc_id":"SOP-PS-002","title":"Patient Safety","section_heading":"Identity Verification","source_path":"sources/SOP-PS-002.html"}]\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "citations",
        data: [
          {
            tag: "S1",
            doc_id: "SOP-PS-002",
            title: "Patient Safety",
            section_heading: "Identity Verification",
            source_path: "sources/SOP-PS-002.html",
          },
        ],
      },
    ]);
  });
});
