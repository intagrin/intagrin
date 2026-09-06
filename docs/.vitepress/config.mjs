import { defineConfig } from 'vitepress'

const SITE_URL = "https://docs.intagrin.com"
const SITE_TITLE = "IntaGrin"
const SITE_DESCRIPTION = "A declarative framework for building multi-agent LLM systems in YAML"
const SOCIAL_IMAGE = `${SITE_URL}/assets/definai_hero_banner.png`

export default defineConfig({
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  base: "/",
  cleanUrls: true,
  sitemap: {
    hostname: SITE_URL
  },
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }],
    ['meta', { name: 'robots', content: 'index, follow' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: SITE_TITLE }],
    ['meta', { property: 'og:image', content: SOCIAL_IMAGE }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: SOCIAL_IMAGE }],
    ['script', { type: 'application/ld+json' }, JSON.stringify({
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      "name": "IntaGrin",
      "applicationCategory": "DeveloperApplication",
      "operatingSystem": "Cross-platform",
      "description": SITE_DESCRIPTION,
      "url": SITE_URL,
      "license": "https://www.apache.org/licenses/LICENSE-2.0",
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
      "codeRepository": "https://github.com/intagrin/intagrin"
    })]
  ],
  // Per-page canonical URL + Open Graph tags derived from each page's own
  // frontmatter description/title, since the static `head` array above only
  // covers site-wide defaults.
  transformHead: ({ pageData }) => {
    const path = pageData.relativePath
      .replace(/index\.md$/, '')
      .replace(/\.md$/, '')
    const url = `${SITE_URL}/${path}`
    const title = pageData.frontmatter?.title || pageData.title
    const ogTitle = title ? `${title} | ${SITE_TITLE}` : SITE_TITLE
    const description = pageData.description || SITE_DESCRIPTION
    return [
      ['link', { rel: 'canonical', href: url }],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:title', content: ogTitle }],
      ['meta', { property: 'og:description', content: description }]
    ]
  },
  themeConfig: {
    logo: "/assets/nav-logo.png",
    siteTitle: "IntaGrin",
    nav: [
      { text: "Documentation", link: "/01_Getting_Started" },
      { text: "GitHub", link: "https://github.com/intagrin/intagrin" }
    ],
    sidebar: [
      {
        text: "🚀 Getting Started",
        items: [
          { text: "Introduction & Philosophy", link: "/01_Getting_Started" },
          { text: "The ai.yaml Blueprint", link: "/02_The_AI_YAML_Blueprint" }
        ]
      },
      {
        text: "🧠 Agents & Workflows",
        items: [
          { text: "Choosing an Orchestration Primitive", link: "/03_Choosing_an_Orchestration_Primitive" },
          { text: "Routing & Handoffs", link: "/03_Agent_Handoffs_and_Routing" },
          { text: "Dynamic Agent Spawning", link: "/03_Dynamic_Agent_Spawning" },
          { text: "Shared Typed State (Redux)", link: "/04_Shared_State_Redux" },
          { text: "Episodic Memory", link: "/14_Episodic_Memory" }
        ]
      },
      {
        text: "🛠️ Tools & Knowledge",
        items: [
          { text: "Custom Tools & Actions", link: "/03_Tools_and_Actions" },
          { text: "Tools & MCP Integration", link: "/05_Custom_Tools_and_MCP" },
          { text: "Advanced RAG & HyDE", link: "/06_Advanced_RAG_and_HyDE" },
          { text: "Agent Skills", link: "/15_Agent_Skills" }
        ]
      },
      {
        text: "🛡️ Enterprise & Safety",
        items: [
          { text: "Human-In-The-Loop (HITL)", link: "/07_Human_In_The_Loop" },
          { text: "Security & Guardrails", link: "/08_Security_and_Reliability" },
          { text: "Security Audit & Threat Model", link: "/08_Security_Audit" },
          { text: "REST API & SSE Streaming", link: "/09_API_and_Streaming" },
          { text: "A2A Interoperability", link: "/16_A2A_Interoperability" }
        ]
      },
      {
        text: "🚢 Deployment & Operations",
        items: [
          { text: "Production Deployment", link: "/04_Production_Deployment" }
        ]
      },
      {
        text: "✨ Developer Experience",
        items: [
          { text: "AI Toolkit & Copilots", link: "/10_AIToolkit_and_Copilots" },
          { text: "IntaGrin Studio", link: "/11_IntaGrin_Studio" },
          { text: "Error Code Reference", link: "/12_Error_Reference" },
          { text: "Configuration Reference", link: "/13_Configuration_Reference" }
        ]
      },
      {
        text: "🏗️ Blueprints",
        items: [
          { text: "Coding Agent", link: "/05_Example_Coding_Agent" },
          { text: "SOC Analyst", link: "/06_Example_SOC_Analyst" },
          { text: "Voice Agent", link: "/07_Example_Voice_Agent" }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/intagrin/intagrin' }
    ],
    footer: {
      message: 'Released under the Apache 2.0 License.',
      copyright: 'Copyright © 2026 IntaGrin'
    },
    search: {
      provider: 'local'
    }
  }
})
