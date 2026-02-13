  # Context

  I'm building a WordPress plugin context service that analyzes 47,000+ plugins and serves structured context to AI agents. When a user asks "Should I use [plugin X]?", the AI agent fetches our
  context to provide a data-informed answer.

  ## The Problem

  Our current AI analysis extracts WHAT a plugin does (features, security, quality) but not WHETHER someone should use it. We're missing guidance like:
  - "Consider this first: Does your host already provide SSL?"
  - "Use this when: You need X but not Y"
  - "Don't use this when: You already have Z"

  ## Current Prompts

  We have 3 analysis agents:

  ### 1. ReadmeSummarizerAgent
  You are a WordPress plugin analyst who extracts structured information from plugin documentation.

  Extract: plugin purpose, features, integrations, target audience, documentation quality.
  Be factual. Focus on what the plugin DOES, not marketing claims.

  Output: summary, primaryPurpose, inferredCategories, keyFeatures, integrations, targetAudience, documentationQuality (0-1), documentationNotes, reasoning

  ### 2. CodeAnalysisAgent
  You are a senior WordPress developer and security expert who reviews plugin code.

  Analyze for:
  - Plugin intent and functionality
  - Code quality and WordPress best practices
  - Security vulnerabilities
  - Extensibility (hooks provided)
  - Performance concerns

  Output: primaryFunction, capabilities, codeQualityScore (0-1), securityScore (0-1), securityFindings[], providedHooks[], performanceConcerns[], reasoning

  ### 3. ChangelogAnalyzerAgent
  You are a technical writer who analyzes software changelogs for actionable information.

  Look for: security fixes, breaking changes, new features, deprecations, update urgency

  Output: hasSecurityFixes, securityFixes[], hasBreakingChanges, breakingChanges[], newFeatures[], updateUrgency (critical/high/medium/low), reasoning

  ## Example Gap

  For "Really Simple SSL", our analysis correctly identifies:
  - 3M installs, 98 rating
  - Security score: 0.6 (medium XSS risk)
  - Features: SSL, 2FA, hardening

  But it DOESN'T tell an AI agent:
  - "Many hosts provide SSL automatically - verify user needs this first"
  - "Originally just SSL, now a full security suite - clarify user's actual need"
  - "Alternatives: manual .htaccess, Cloudflare, host-provided SSL"

  ## Your Task

  1. Critique our current prompts - what's working, what's missing?

  2. Propose specific and explicit improvements to the prompts.

  3. Determine if new prompt is needed, one that generates "recommendation guidance" including:
     - whenToUse / whenNotToUse
     - considerFirst (questions to ask before recommending)
     - alternatives (other solutions to the same problem)
     - caveats (things to know even if you use it)

  4. Should this be a 4th agent or integrated into existing prompts?

  5. Be clear on what to do: we already have a system that uses these prompts so I need to know exactly what to do.

  6. How do we handle plugins where we DON'T have good guidance (long tail)?

  Be specific and provide example output for "Really Simple SSL".
