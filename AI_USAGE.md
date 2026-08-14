# AI Usage

I used **ChatGPT** and **Claude** as development assistants throughout the project. I used them primarily for debugging, code review, architecture discussions, test/evaluation design, and identifying likely failure modes. I did not treat generated code as authoritative; I tested changes against the application, inspected logs/network traces, and changed or rejected suggestions when they did not match the actual system behavior.

## Examples of overrides and corrections

1. **HTML parsing / chunking.** AI-generated parsing logic was more complicated than necessary. I changed the implementation myself to use **Beautiful Soup with a DOM-based extractor**, which made the HTML parsing logic substantially simpler and more maintainable for the actual corpus.

2. **Embedding efficiency.** The initial implementation generated embeddings one chunk at a time:

   ```python
   embeddings = [embedder.embed_document(c.metadata.text) for c in chunks]
   ```

   I identified this as an API-efficiency problem because one chunk resulted in one embedding request. I deliberately kept this as a known limitation for the time box but identified the required production fix: batch embedding requests, pass output dimensionality through the embedding call, and validate the returned dimensionality before indexing.

3. **SSE/frontend debugging.** AI initially focused on backend generation/token limits when the UI was not displaying responses. I independently inspected the browser Network tab and found that the backend was successfully streaming complete answers. I then traced the frontend parser and corrected a format mismatch: the backend sent `data: {"type":"token",...}` while the frontend was incorrectly waiting for a separate `event:` field. This was a direct correction based on observed wire data rather than accepting the initial diagnosis.

4. **Evaluation dataset.** The first eval dataset used expected source IDs such as `SOP-PS-002`, but the actual Pinecone corpus contained IDs such as `cdc-standard-precautions`. I rejected the original mapping rather than accepting a misleading retrieval score and rewrote the answerable questions to correspond to the actual indexed corpus and canonical `document_id` values.

5. **Evaluation error handling and API limits.** The initial eval harness treated a generation failure as if the application had produced an `answered` result, which would have polluted status accuracy. I changed the evaluation model so API/infrastructure failures are represented as evaluation errors and excluded from status accuracy while still being reported. I also reduced the suite to the required 15 cases and removed duplicate retrieval work to control API consumption.

6. **Classifier behavior.** I did not remove the medical-advice classifier from the eval simply to save API calls. The assignment requires testing that the app actually declines personal medical advice, so I kept the classifier in the live/eval path and instead narrowed when the LLM classifier is invoked, using high-confidence deterministic patterns first.

The most important pattern in my AI usage was **verification**: I used AI to accelerate implementation and reasoning, but I relied on application behavior, logs, network traces, API responses, and the actual indexed data to decide whether a proposed change was correct.
