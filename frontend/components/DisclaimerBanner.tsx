import styles from "./DisclaimerBanner.module.css";

export default function DisclaimerBanner() {
  return (
    <aside className={styles.banner} role="note">
      <div className={styles.label}>REFERENCE USE</div>

      <div className={styles.content}>
        <strong>Clinical operations reference tool.</strong>
        <span>
          This tool provides information from the organization&apos;s indexed
          reference documents. It is not a substitute for professional medical
          judgment or patient-specific medical advice.
        </span>
      </div>
    </aside>
  );
}
