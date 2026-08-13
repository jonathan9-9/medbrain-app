"use client";

import styles from "./ExamplePrompts.module.css";

interface ExamplePromptsProps {
  onSelect: (prompt: string) => void;
  disabled?: boolean;
}

const PROMPTS = [
  "What two identifiers must staff use to verify a patient's identity before giving a medication?",
  "What temperature range should refrigerated vaccines be kept at?",
  "How many emergency response drills is each site required to run per year?",
  "What is the correct order for removing PPE?",
];

export default function ExamplePrompts({
  onSelect,
  disabled = false,
}: ExamplePromptsProps) {
  return (
    <section
      className={styles.container}
      aria-labelledby="example-prompts-title"
    >
      <div className={styles.header}>
        <span className={styles.label}>QUICK REFERENCE</span>
        <h2 id="example-prompts-title">Example questions</h2>
      </div>

      <div className={styles.grid}>
        {PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className={styles.prompt}
            onClick={() => onSelect(prompt)}
            disabled={disabled}
          >
            <span className={styles.arrow} aria-hidden="true">
              →
            </span>
            <span>{prompt}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
