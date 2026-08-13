import type { AnswerStatus } from "@/lib/types";
import styles from "./StatusIndicator.module.css";

interface StatusIndicatorProps {
  status: AnswerStatus;
}

const STATUS_LABELS: Record<AnswerStatus, string> = {
  answered: "Answered",
  unanswerable: "Unable to answer",
  refused_medical_advice: "Medical advice request declined",
};

export default function StatusIndicator({ status }: StatusIndicatorProps) {
  return (
    <div
      className={`${styles.indicator} ${styles[status]}`}
      role="status"
      aria-label={`Response status: ${STATUS_LABELS[status]}`}
    >
      <span className={styles.dot} aria-hidden="true" />
      <span>{STATUS_LABELS[status]}</span>
    </div>
  );
}
