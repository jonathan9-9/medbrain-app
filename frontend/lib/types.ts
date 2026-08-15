export type AnswerStatus =
  | "answered"
  | "unanswerable"
  | "refused_medical_advice";

export interface Citation {
  tag: string;
  doc_id: string;
  title: string;
  section_heading: string;
  source_path: string;
}

export type ChatMessageEventType =
  | "token"
  | "citations"
  | "status"
  | "retrieval"
  | "error"
  | "done";

export interface ChatMessageEvent {
  type: ChatMessageEventType;
  data: string | Citation[] | string[] | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  status?: AnswerStatus;
}
