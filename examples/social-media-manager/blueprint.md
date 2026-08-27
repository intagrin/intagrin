# Product Blueprint: Social Media Manager AI

## 1. Product Vision
The Social Media Manager AI is an autonomous multi-agent system designed to streamline the identification, creation, review, and publishing lifecycle for professional content. Focused primarily on AI and IT trends, the system leverages live web research, multi-stage drafting, and rigorous quality and human-in-the-loop (HITL) review processes to deliver high-impact LinkedIn posts efficiently.

---

## 2. Technical Architecture & Routing

The system operates via a collaborative multi-agent architecture orchestrated through defined handoffs and a structured execution workflow.


                  +-----------------------+
                  |    research_agent     |
                  | (fetch_trending_ai...) |
                  +-----------------------+
                     |                 ^
          (handoff)  |                 |  (handoff)
                     v                 |
                  +-----------------------+
                  |content_generator_agent|
                  | (search_topic_details)|
                  +-----------------------+
                     |                 ^
          (handoff)  |                 |  (handoff)
                     v                 |
                  +-----------------------+
                  |     reviewer_agent    |
                  |   (post_to_linkedin)  |
                  +-----------------------+


### Default Agent & Models
* **Default Entry Agent:** `research_agent`
* **Primary Model:** `gemini/gemini-3.5-flash-lite`
* **Fallback Model:** `gemini/gemini-3.5-flash`
* **Temperature:** `0.7`
* **Guardrails:** PII Masking enabled (`mask_pii: true`)
* **Circuit Breakers:** Maximum of 10 handoffs per session

---

## 3. Agent Specifications & Tools

### A. Research Agent (`research_agent`)
* **Description:** Identifies and researches trending AI and IT topics from live web sources.
* **System Prompt:** `prompts/research_agent.jinja2`
* **Allowed Handoffs:** `content_generator_agent`, `reviewer_agent`
* **Tools:**
  * `fetch_trending_ai_topics` (`tools.custom_tools.fetch_trending_ai_topics`)

### B. Content Generator Agent (`content_generator_agent`)
* **Description:** Conducts deep web research on the chosen topic and drafts engaging LinkedIn posts.
* **System Prompt:** `prompts/content_generator_agent.jinja2`
* **Allowed Handoffs:** `reviewer_agent`, `research_agent`
* **Tools:**
  * `search_topic_details` (`tools.custom_tools.search_topic_details`)

### C. Reviewer Agent (`reviewer_agent`)
* **Description:** Reviews post quality, clarity, and tone, then publishes approved content to LinkedIn.
* **System Prompt:** `prompts/reviewer_agent.jinja2`
* **Allowed Handoffs:** `content_generator_agent`, `research_agent`
* **Tools:**
  * `post_to_linkedin` (`tools.custom_tools.post_to_linkedin`) — *Requires Approval*

---

## 4. Execution Workflow

The system follows a flexible multi-agent pipeline to take a topic from concept to publication:

1. **Research Topic (`research_agent`):**
   * Explore the latest trending topics in AI and IT using live web sources.
   * Hand off findings to the content generator or directly to the reviewer if necessary.
2. **Generate LinkedIn Post (`content_generator_agent`):**
   * Conduct deep web research on the chosen topic details.
   * Draft a polished, high-engagement LinkedIn post.
3. **Review and Publish (`reviewer_agent`):**
   * Review post quality, clarity, and tone.
   * Secure necessary approvals and execute publication to LinkedIn using the `post_to_linkedin` tool.

---

## 5. Memory & Telemetry

* **Memory Store:** SQLite (`.ai/memory.db`) persistent session memory.
* **Telemetry:** OpenTelemetry (`otel`) integrated for observability and tracing across agent steps, tool executions, and handoffs.