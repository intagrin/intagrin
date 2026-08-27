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
