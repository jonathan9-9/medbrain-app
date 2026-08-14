import { describe, expect, it } from "vitest";
import { parseSseBuffer } from "@/lib/api";

describe("parseSseBuffer", () => {
  it("parses the backend's JSON-encoded status event", () => {
    const result = parseSseBuffer(
      'data: {"type":"status","data":"answered"}\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "status",
        data: "answered",
      },
    ]);
    expect(result.remaining).toBe("");
  });

  it("parses a streamed token event", () => {
    const result = parseSseBuffer('data: {"type":"token","data":"Hello"}\n\n');

    expect(result.events).toEqual([
      {
        type: "token",
        data: "Hello",
      },
    ]);
  });

  it("handles multiple backend events in one buffer", () => {
    const result = parseSseBuffer(
      'data: {"type":"status","data":"answered"}\n\n' +
        'data: {"type":"token","data":"Hello"}\n\n' +
        'data: {"type":"done","data":null}\n\n',
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
      {
        type: "done",
        data: null,
      },
    ]);
    expect(result.remaining).toBe("");
  });

  it("preserves an incomplete event across chunk boundaries", () => {
    const first = parseSseBuffer('data: {"type":"token","data":"Hello"}');

    expect(first.events).toEqual([]);
    expect(first.remaining).toBe('data: {"type":"token","data":"Hello"}');

    const second = parseSseBuffer(`${first.remaining}\n\n`);

    expect(second.events).toEqual([
      {
        type: "token",
        data: "Hello",
      },
    ]);
    expect(second.remaining).toBe("");
  });

  it("parses retrieval document IDs", () => {
    const result = parseSseBuffer(
      'data: {"type":"retrieval","data":["cdc-standard-precautions","cdc-hand-hygiene"]}\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "retrieval",
        data: ["cdc-standard-precautions", "cdc-hand-hygiene"],
      },
    ]);
  });

  it("parses citation arrays", () => {
    const result = parseSseBuffer(
      'data: {"type":"citations","data":[{"tag":"S1","doc_id":"cdc-standard-precautions","title":"Standard Precautions for All Patient Care","section_heading":"Standard Precautions","source_path":"/corpus/raw/cdc_standard_precautions.html"}]}\n\n',
    );

    expect(result.events).toEqual([
      {
        type: "citations",
        data: [
          {
            tag: "S1",
            doc_id: "cdc-standard-precautions",
            title: "Standard Precautions for All Patient Care",
            section_heading: "Standard Precautions",
            source_path: "/corpus/raw/cdc_standard_precautions.html",
          },
        ],
      },
    ]);
  });

  it("ignores malformed JSON", () => {
    const result = parseSseBuffer("data: this-is-not-json\n\n");

    expect(result.events).toEqual([]);
    expect(result.remaining).toBe("");
  });
});
