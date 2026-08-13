import DisclaimerBanner from "@/components/DisclaimerBanner";
import CitationCard from "@/components/CitationCard";
import MessageBubble from "@/components/MessageBubble";
import StatusIndicator from "@/components/StatusIndicator";

export default function HomePage() {
  return (
    <main>
      <DisclaimerBanner />
      <MessageBubble />
      <StatusIndicator />
      <CitationCard />
    </main>
  );
}
