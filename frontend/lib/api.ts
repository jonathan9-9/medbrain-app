import type { ChatMessageEvent } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

export function parseSseBuffer(buffer: string): {
  events: ChatMessageEvent[];
  remaining: string;
} {
  const events: ChatMessageEvent[] = [];

  const parts = buffer.split("\n\n");
  const remaining = parts.pop() ?? "";

  for (const part of parts) {
    const lines = part.split("\n");

    let data = "";

    for (const line of lines) {
      if (line.startsWith("data:")) {
        const value = line.slice("data:".length).trim();

        if (data) {
          data += "\n";
        }

        data += value;
      }
    }

    if (!data) {
      continue;
    }

    try {
      const parsed = JSON.parse(data);

      if (
        typeof parsed === "object" &&
        parsed !== null &&
        "type" in parsed &&
        "data" in parsed
      ) {
        events.push({
          type: parsed.type as ChatMessageEvent["type"],
          data: parsed.data as ChatMessageEvent["data"],
        });
      }
    } catch {
      continue;
    }
  }

  return {
    events,
    remaining,
  };
}

export async function streamChat(
  question: string,
  onEvent: (event: ChatMessageEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed with status ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Chat response did not contain a readable stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      const parsed = parseSseBuffer(buffer);

      buffer = parsed.remaining;

      for (const event of parsed.events) {
        onEvent(event);
      }
    }

    // Flush any remaining decoder bytes.
    buffer += decoder.decode();

    const parsed = parseSseBuffer(`${buffer}\n\n`);

    for (const event of parsed.events) {
      onEvent(event);
    }
  } finally {
    reader.releaseLock();
  }
}
