import type { Citation } from "@/lib/types";
import styles from "./CitationCard.module.css";

interface CitationCardProps {
  citation: Citation;
}

export default function CitationCard({ citation }: CitationCardProps) {
  return (
    <article className={styles.card}>
      <div className={styles.tag}>{citation.tag}</div>

      <div className={styles.details}>
        <div className={styles.title}>{citation.title}</div>

        <div className={styles.section}>{citation.section_heading}</div>

        <div className={styles.docId}>{citation.doc_id}</div>
      </div>
    </article>
  );
}
