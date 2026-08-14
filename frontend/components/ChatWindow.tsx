"use client";

import { useRef, useState } from "react";

import { streamChat } from "@/lib/api";
import type {
  AnswerStatus,
  ChatMessage,
  ChatMessageEvent,
  Citation,
} from "@/lib/types";

import ChatInput from "./ChatInput";
import CitationCard from "./CitationCard";
import ExamplePrompts from "./ExamplePrompts";
import MessageBubble from "./MessageBubble";
import StatusIndicator from "./StatusIndicator";

import styles from "./ChatWindow.module.css";

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<AnswerStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  async function handleSubmit(question: string) {
    if (isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      citations: [],
    };

    const assistantMessageId = crypto.randomUUID();

    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      citations: [],
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);

    setStatus(null);
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChat(
        question,
        (event: ChatMessageEvent) => {
          switch (event.type) {
            case "token": {
              if (typeof event.data !== "string") {
                return;
              }

              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        content: message.content + event.data,
                      }
                    : message,
                ),
              );

              break;
            }

            case "status": {
              if (
                event.data !== "answered" &&
                event.data !== "unanswerable" &&
                event.data !== "refused_medical_advice"
              ) {
                return;
              }

              const nextStatus: AnswerStatus = event.data;

              setStatus(nextStatus);

              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        status: nextStatus,
                      }
                    : message,
                ),
              );

              break;
            }

            case "citations": {
              if (!Array.isArray(event.data)) {
                return;
              }

              const citations = event.data.filter(
                (item): item is Citation =>
                  typeof item === "object" &&
                  item !== null &&
                  "tag" in item &&
                  "doc_id" in item &&
                  "title" in item &&
                  "section_heading" in item &&
                  "source_path" in item,
              );

              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        citations,
                      }
                    : message,
                ),
              );

              break;
            }

            case "retrieval": {
              // Retrieval events are useful for evaluation/debugging,
              // but raw document IDs are not displayed to the user.
              break;
            }

            case "error": {
              console.error("Chat stream error:", event.data);
              break;
            }

            case "done": {
              setIsLoading(false);
              break;
            }

            default:
              break;
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setIsLoading(false);

        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content: message.content || "Response interrupted by user.",
                }
              : message,
          ),
        );

        return;
      }
    } finally {
      abortControllerRef.current = null;
    }
  }

  function handleExamplePrompt(prompt: string) {
    void handleSubmit(prompt);
  }

  return (
    <section className={styles.container}>
      <header className={styles.header}>
        <div>
          <span className={styles.eyebrow}>CLINICAL OPERATIONS</span>

          <h1>Reference Desk</h1>

          <p>
            Search the indexed clinical operations documents using natural
            language.
          </p>
        </div>

        {isLoading && (
          <button
            type="button"
            className={styles.stopButton}
            onClick={() => abortControllerRef.current?.abort()}
          >
            Stop
          </button>
        )}
      </header>

      <div className={styles.transcript}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyMarker}>REF</div>

            <h2>Clinical operations reference</h2>

            <p>
              Ask about policies, procedures, timelines, safety requirements, or
              other information contained in the indexed document set.
            </p>

            <ExamplePrompts
              onSelect={handleExamplePrompt}
              disabled={isLoading}
            />
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={styles.messageRow}>
                <MessageBubble message={message} />

                {message.role === "assistant" &&
                  message.citations.length > 0 && (
                    <div className={styles.citations}>
                      <div className={styles.citationsHeader}>
                        <span>DOCUMENT SOURCES</span>
                      </div>

                      <div className={styles.citationList}>
                        {message.citations.map((citation) => (
                          <CitationCard
                            key={`${citation.tag}-${citation.doc_id}-${citation.section_heading}`}
                            citation={citation}
                          />
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            ))}

            {status && <StatusIndicator status={status} />}
          </>
        )}
      </div>

      <div className={styles.inputArea}>
        {messages.length === 0 && (
          <div className={styles.inputLabel}>
            <span>QUERY</span>
          </div>
        )}

        <ChatInput onSubmit={handleSubmit} disabled={isLoading} />
      </div>
    </section>
  );
}
