import type { ChatMessage } from "@/lib/types";
import CitationCard from "./CitationCard";
import StatusIndicator from "./StatusIndicator";
import styles from "./MessageBubble.module.css";

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <article
      className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}
    >
      <div className={styles.meta}>
        <span className={styles.role}>{isUser ? "YOU" : "REFERENCE DESK"}</span>

        {!isUser && message.status && (
          <StatusIndicator status={message.status} />
        )}
      </div>

      <div className={styles.content}>
        {message.content || (
          <span className={styles.waiting}>Generating response...</span>
        )}
      </div>

      {!isUser && message.citations.length > 0 && (
        <div className={styles.citations}>
          {message.citations.map((citation) => (
            <CitationCard
              key={`${citation.tag}-${citation.doc_id}`}
              citation={citation}
            />
          ))}
        </div>
      )}
    </article>
  );
}
