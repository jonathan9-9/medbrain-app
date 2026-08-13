"use client";

import { FormEvent, useState } from "react";
import styles from "./ChatInput.module.css";

interface ChatInputProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
}

export default function ChatInput({
  onSubmit,
  disabled = false,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const question = value.trim();

    if (!question || disabled) {
      return;
    }

    onSubmit(question);
    setValue("");
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <div className={styles.inputWrapper}>
        <textarea
          className={styles.input}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Ask a question about clinical operations..."
          disabled={disabled}
          rows={2}
          maxLength={2000}
          aria-label="Clinical operations question"
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />

        <div className={styles.footer}>
          <span className={styles.hint}>
            Press Enter to send · Shift + Enter for a new line
          </span>

          <span className={styles.counter}>{value.length}/2000</span>

          <button
            type="submit"
            className={styles.button}
            disabled={disabled || !value.trim()}
          >
            {disabled ? "Searching..." : "Ask"}
          </button>
        </div>
      </div>
    </form>
  );
}
