---
layout: home

hero:
  name: "IntaGrin"
  text: "A declarative framework for building multi-agent LLM systems in YAML"
  tagline: Describe agents, handoffs, tools, and guardrails in ai.yaml — verified before you run them, not just prompted and hoped for.
  image:
    light: /assets/logo2.png
    dark: /assets/logo3.png
    alt: IntaGrin
  actions:
    - theme: brand
      text: Get Started
      link: /01_Getting_Started
    - theme: alt
      text: View on GitHub
      link: https://github.com/intagrin/intagrin

features:
  - icon: 🎯
    title: Declarative, not programmatic
    details: Routing, guardrails, and memory live in ai.yaml, not hand-wired Python graph code or agent/crew objects.
  - icon: 🛡️
    title: Governance built in, not bolted on
    details: Circuit breakers with an honest bounded/unbounded cost accounting, per-call human-in-the-loop approval, content-provenance guardrails, and sandboxed code execution.
  - icon: ✅
    title: Static verification before runtime
    details: inta verify catches routing cycles, unbounded cost paths, and misconfigured guardrails before you ever run the agent.
  - icon: 📖
    title: Honesty over marketing
    details: Every tool in this repo says what it doesn't cover, not just what it does — a claim only ships if the code actually backs it.
---

<div class="landing-section story-section">

<div class="story-text">

## The whole idea, in under a minute

No hand-wired routing, no loops nobody planned, no token bill nobody bounded. Declare the swarm
instead: one command turns it into a running graph, and every agent loads only the tools it
actually needs — smaller prompt, smaller bill. Same commands shown throughout this page, just
faster to watch than to read.

</div>

<div class="story-video-wrap">

<video class="story-video" autoplay muted loop playsinline data-webm-light="/assets/story.webm" data-mp4-light="/assets/story.mp4" data-webm-dark="/assets/story-dark.webm" data-mp4-dark="/assets/story-dark.mp4" aria-label="A short animated walkthrough of IntaGrin: declaring a swarm, compiling it, watching it run, and loading only the tools each agent needs" />

</div>

</div>

<div class="landing-section">

## Spec-driven, not vibe-coded

Describe what you want in plain English — `blueprint.md` — and let `inta compile` turn it into a
validated `ai.yaml`. Re-running it against an updated blueprint diffs against what you already
have and merges instead of overwriting, self-healing against schema and router-condition errors
before anything ever touches disk.

<div class="blueprint-flow">

<div class="code-panel">
<div class="code-panel-title">blueprint.md</div>

```markdown
# Vision
A support agent that triages tickets, hands
billing to a specialist, and asks a human
before issuing any refund.
```

</div>

<div class="code-panel">
<div class="code-panel-title">terminal</div>

```bash
$ inta compile blueprint.md

✓ Schema + router-condition validated
✓ Scaffolded prompts/triage.jinja2
✓ inta verify: acyclic, cost-bounded

Successfully compiled blueprint into ai.yaml
```

</div>

</div>
</div>

<div class="landing-section">

## See everything. Approve anything.

A live, drag-and-drop graph of your actual `ai.yaml` — not a diagram you drew once and forgot to
update. Watch handoffs happen in real time, review exactly which arguments a gated tool call wants
to run with before you approve it — including a generated image, not just a wall of JSON — and
push a visual edit straight back to `ai.yaml` on disk. `inta monitor` — nothing to deploy, nothing
to sign up for.

<video class="dashboard-shot" autoplay muted loop playsinline poster="/assets/monitor-dashboard.png" aria-label="A recorded walkthrough of the IntaGrin Monitor dashboard: the live agent graph, the Playground, and a human-in-the-loop approval with its actual arguments">
  <source src="/assets/monitor-demo.webm" type="video/webm">
  <source src="/assets/monitor-demo.mp4" type="video/mp4">
</video>

</div>

