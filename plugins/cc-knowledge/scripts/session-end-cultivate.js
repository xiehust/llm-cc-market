#!/usr/bin/env node

const path = require('path');
const fs = require('fs');
const { spawn, spawnSync } = require('child_process');
const {
  analyzeSession,
  MIN_SESSION_SCORE,
  getWikiHub,
  ensureDir,
  writeFile,
  readFile,
  getPendingDir,
  getDateString,
  log
} = require('./lib/utils');

const MIN_MESSAGES = 8;
const MAX_LESSONS_PER_SESSION = 5;
const TOMBSTONE_TTL_DAYS = 7;

function readStdinJson() {
  return new Promise((resolve) => {
    // If stdin is a TTY there is no piped payload — resolve empty immediately.
    if (process.stdin.isTTY) {
      resolve({});
      return;
    }
    let data = '';
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      try {
        resolve(data.trim() ? JSON.parse(data) : {});
      } catch {
        resolve({});
      }
    };
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', finish);
    process.stdin.on('error', finish);
    // Safety net: never hang the SessionEnd hook waiting on stdin.
    setTimeout(finish, 2000).unref();
  });
}

async function main() {
  // Claude Code delivers hook payload as JSON on stdin (NOT env vars).
  // Fall back to CLAUDE_CODE_SESSION_ID env for the session id when present.
  const input = await readStdinJson();
  const transcriptPath = input.transcript_path || process.env.CLAUDE_TRANSCRIPT_PATH;
  const sessionId =
    input.session_id ||
    process.env.CLAUDE_CODE_SESSION_ID ||
    process.env.CLAUDE_SESSION_ID ||
    'unknown';

  if (!transcriptPath || !fs.existsSync(transcriptPath)) {
    process.exit(0);
  }

  // Per-session dedup: skip if this session was cultivated in the last N days
  if (isSessionRecentlyProcessed(sessionId)) {
    log(`[cc-knowledge] Session ${sessionId.slice(-12)} already cultivated within ${TOMBSTONE_TTL_DAYS}d — skipping`);
    process.exit(0);
  }

  // Stage 1 — cheap weighted heuristic gate (zero tokens).
  // A lone trivial edit no longer qualifies; we require real signal
  // (error→fix, user corrections, multi-file work) to clear the threshold.
  const signals = analyzeSession(transcriptPath);
  if (signals.msgCount < MIN_MESSAGES) {
    process.exit(0);
  }
  if (signals.score < MIN_SESSION_SCORE) {
    log(`[cc-knowledge] Session below cultivation threshold (score=${signals.score} < ${MIN_SESSION_SCORE}) — skipping`);
    process.exit(0);
  }

  const topicHint = extractTopicHint(transcriptPath);
  const hubPath = getWikiHub();

  log(`[cc-knowledge] Session passed heuristic gate (score=${signals.score}: ${signals.editCount} edits across ${signals.distinctFiles} files, ${signals.errorFixCount} error→fix, ${signals.correctionCount} corrections)`);

  // Stage 2 — cheap LLM triage (haiku) before committing to a full opus
  // extraction. Asks a single yes/no: does ≥1 durable, transferable lesson
  // plausibly exist here? Fails OPEN (proceeds) on any error so we never
  // silently drop a session because triage was unavailable.
  const claudePath = findClaude();
  if (claudePath && !shouldCultivate(claudePath, transcriptPath, signals)) {
    log('[cc-knowledge] Triage (haiku) judged session not worth extracting — skipping');
    process.exit(0);
  }

  log(`[cc-knowledge] Topic hint: "${topicHint}" | Hub: ${hubPath}`);

  // Write pending marker
  const pendingDir = getPendingDir();
  ensureDir(pendingDir);
  const marker = {
    sessionId,
    topicHint,
    hubPath,
    transcriptPath,
    timestamp: new Date().toISOString(),
    date: getDateString(),
    gateSignals: signals,
    status: 'pending'
  };

  const markerPath = path.join(pendingDir, `${sessionId.slice(-12)}.json`);

  // Try to spawn claude for automatic extraction (claudePath resolved above).
  if (claudePath) {
    const skillPath = path.join(__dirname, '..', 'skills', 'cultivator-engine', 'SKILL.md');
    if (fs.existsSync(skillPath)) {
      marker.status = 'spawned';
      writeFile(markerPath, JSON.stringify(marker, null, 2));

      // Stamp tombstone before spawn — prevents re-fire even if the cultivator silently exits.
      // Cultivator deletes the marker on success; we keep the tombstone separately.
      writeSessionTombstone(sessionId);

      const prompt = buildExtractionPrompt(hubPath, markerPath, topicHint);
      // Use --resume to inherit full session context (avoids passing a lossy excerpt).
      const child = spawn(
        claudePath,
        ['-p', '--resume', sessionId, '--model', 'claude-opus-4-8[1m]', prompt],
        {
          detached: true,
          stdio: 'ignore',
          env: { ...process.env, CC_KNOWLEDGE_MARKER: markerPath }
        }
      );
      child.unref();

      log(`[cc-knowledge] Spawned cultivator (pid ${child.pid}) — resumed session ${sessionId.slice(-12)}`);
      process.exit(0);
    }
  }

  // Fallback: deferred cultivation
  marker.status = 'deferred';
  writeFile(markerPath, JSON.stringify(marker, null, 2));
  log('[cc-knowledge] Cultivation deferred — run /cc-knowledge:cultivate next session');
  process.exit(0);
}

function getTombstoneDir() {
  return path.join(getPendingDir(), 'processed-sessions');
}

function isSessionRecentlyProcessed(sessionId) {
  if (!sessionId || sessionId === 'unknown') return false;
  const tombstone = path.join(getTombstoneDir(), `${sessionId}.json`);
  if (!fs.existsSync(tombstone)) return false;
  try {
    const stat = fs.statSync(tombstone);
    const ageMs = Date.now() - stat.mtimeMs;
    return ageMs < TOMBSTONE_TTL_DAYS * 24 * 60 * 60 * 1000;
  } catch {
    return false;
  }
}

function writeSessionTombstone(sessionId) {
  if (!sessionId || sessionId === 'unknown') return;
  const dir = getTombstoneDir();
  ensureDir(dir);
  writeFile(
    path.join(dir, `${sessionId}.json`),
    JSON.stringify({ sessionId, processedAt: new Date().toISOString() }, null, 2)
  );
}

function extractTopicHint(transcriptPath) {
  try {
    const content = fs.readFileSync(transcriptPath, 'utf8');
    const lines = content.split('\n').filter(Boolean);
    const userMessages = [];
    for (const line of lines.slice(-40)) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'user' && typeof entry.content === 'string') {
          userMessages.push(entry.content);
        } else if (entry.type === 'user' && Array.isArray(entry.content)) {
          for (const block of entry.content) {
            if (block.type === 'text') userMessages.push(block.text);
          }
        }
      } catch {}
    }
    const lastMsgs = userMessages.slice(-5).join(' ');
    const words = lastMsgs
      .replace(/[^a-zA-Z0-9一-鿿\s-]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 2)
      .slice(0, 5);
    return words.join(' ') || 'general';
  } catch {
    return 'general';
  }
}

function findClaude() {
  const result = spawnSync('which', ['claude'], { encoding: 'utf8', stdio: 'pipe' });
  if (result.status === 0 && result.stdout.trim()) {
    return result.stdout.trim();
  }
  const commonPaths = [
    path.join(getHomeDir(), '.claude', 'local', 'claude'),
    '/usr/local/bin/claude',
    '/opt/homebrew/bin/claude'
  ];
  for (const p of commonPaths) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function getHomeDir() {
  return require('os').homedir();
}

// Cheap LLM triage: ask haiku whether a durable, transferable lesson plausibly
// exists. Synchronous + short timeout so it doesn't stall the SessionEnd hook.
// Returns true (cultivate) on ANY uncertainty or error — fail open.
function shouldCultivate(claudePath, transcriptPath, signals) {
  let excerpt;
  try {
    excerpt = buildTriageExcerpt(transcriptPath);
  } catch {
    return true;
  }
  if (!excerpt) return true;

  const prompt = buildTriagePrompt(excerpt, signals);
  try {
    const result = spawnSync(
      claudePath,
      ['-p', '--model', 'claude-haiku-4-5-20251001', prompt],
      {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 30000,
        maxBuffer: 1024 * 1024
      }
    );
    if (result.status !== 0 || !result.stdout) return true; // fail open
    // Look at the first decisive token. Only an explicit NO skips cultivation.
    const verdict = result.stdout.trim().toUpperCase();
    if (/^\s*NO\b/.test(verdict) || /\bVERDICT\s*[:=]?\s*NO\b/.test(verdict)) {
      return false;
    }
    return true;
  } catch {
    return true; // fail open on timeout / spawn error
  }
}

// Build a compact, token-bounded excerpt of user turns + error/correction
// context for the triage model. We avoid resuming the full session here —
// that's the expensive step we're trying to gate.
function buildTriageExcerpt(transcriptPath, maxChars = 6000) {
  const content = readFile(transcriptPath);
  if (!content) return '';
  const lines = content.split('\n').filter(Boolean);
  const picked = [];
  for (const line of lines) {
    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (entry.type !== 'user') continue;
    let text = '';
    if (typeof entry.content === 'string') {
      text = entry.content;
    } else if (Array.isArray(entry.content)) {
      text = entry.content
        .filter(b => b && b.type === 'text' && typeof b.text === 'string')
        .map(b => b.text)
        .join(' ');
    }
    text = text.trim();
    // Skip tool-result-only user turns and noise.
    if (!text || text.startsWith('<')) continue;
    picked.push(text.replace(/\s+/g, ' ').slice(0, 500));
  }
  // Keep the most recent turns, bounded by maxChars.
  const recent = [];
  let total = 0;
  for (let i = picked.length - 1; i >= 0; i--) {
    if (total + picked[i].length > maxChars) break;
    recent.unshift(picked[i]);
    total += picked[i].length;
  }
  return recent.map(t => `- ${t}`).join('\n');
}

function buildTriagePrompt(excerpt, signals) {
  return `You are a strict triage filter for a knowledge wiki. Decide whether a Claude Code session is worth mining for a DURABLE, TRANSFERABLE lesson (a non-obvious gotcha, debugging insight, design tradeoff, or reusable pattern that generalizes beyond this one project).

Heuristic signals already detected: ${signals.errorFixCount} error→fix sequence(s), ${signals.correctionCount} user correction(s), edits across ${signals.distinctFiles} file(s).

User turns from the session (most recent last):
${excerpt}

Answer with exactly one word on the first line: YES if at least one durable transferable lesson plausibly exists, NO otherwise. Bias toward NO for routine activity, simple Q&A, happy-path edits, or project-specific trivia with no transferable rule.`;
}

function buildExtractionPrompt(hubPath, markerPath, topicHint) {
  const date = getDateString();
  return `You are extracting durable lessons from this Claude Code session.
The full session transcript is loaded as your context (you've been resumed into it) — do NOT ask for an excerpt.

Hub: ${hubPath}
Marker to delete on exit: ${markerPath}
Topic hint (advisory only): "${topicHint}"
Today: ${date}

# Hard rules — disobeying any of these wastes the user's wiki

## Rule 0 — Default is "write nothing"

If <2 lessons survive the filters below, **delete the marker and exit silently. Write no file.**
Do NOT write a "no new lessons" / "dedup" / "re-confirmed" meta-note. Those notes are themselves the problem we are fixing — they pollute the wiki and trigger false dedup loops on future runs.

## Rule 1 — A lesson must be DURABLE

A candidate qualifies as a lesson only if ALL hold:
- Encodes a non-obvious rule, gotcha, or pattern future-you would want to remember
- Generalizes beyond this specific project / file / branch
- Has a concrete failure mode, surprising behavior, or non-trivial design tradeoff at its core

Reject categorically:
- Activity summaries / "what we did today" / progress reports
- Meta-notes about extraction state (dedup notes, "no new lessons", re-confirmation logs)
- Project-internal trivia ("file X needs flag Y in repo Z") with no transferable rule
- Things obvious from reading the code or framework's own docs
- Successful happy-path narratives without a surprising step

## Rule 2 — Dedup against existing notes

Before writing, list active ${hubPath}/topics/*/raw/notes/ files modified in the last 14 days and skim their Rule lines and titles. Skip ${hubPath}/topics/.archive/ unless an explicit archived write was requested. For each candidate lesson, drop it if its Rule line is already covered (even with different wording). If all candidates are duplicates → exit silently per Rule 0.

## Rule 3 — Topic targeting (no lazy "general")

1. Read ${hubPath}/wikis.json to enumerate topics.
2. Ignore topics with status "archived" or paths under topics/.archive/ unless an explicit archived write was requested.
3. Pick the most specific active topic whose description matches the lesson's domain.
4. Use "general" ONLY if the lesson is genuinely cross-domain AND no specific active topic fits. When torn between specific and general, choose specific.
5. If no active topic fits and the lesson is high-value enough to justify a new topic, create one (see ${hubPath}/AGENTS.md or follow the cultivator-engine skill).

## Rule 4 — Cap and shape

- Max ${MAX_LESSONS_PER_SESSION} lessons per session. Quality over quantity — 2 sharp lessons beat 5 mushy ones.
- Each lesson body: Category (gotcha|pattern|rule|discovery|correction), Context, Symptom, Root cause, Fix, Rule (one generalizable sentence).
- Be specific: include exact error strings, file paths, tool/library names, version numbers when relevant.

# Workflow

1. Scan the resumed session for error→fix sequences, user corrections, gotchas, undocumented behaviors.
2. Apply Rules 1–4 (durable filter → dedup → topic selection).
3. If <2 lessons survive → delete ${markerPath} and exit silently. STOP.
4. Otherwise: pick the chosen active topic, write to ${hubPath}/topics/<chosen-topic>/raw/notes/${date}-ll-<descriptive-slug>.md with llm-wiki frontmatter (type: notes, source: "session", ingested: ${date}, tags: [...specific tags], confidence: high, summary: <one sentence>).
5. Update the topic's raw/notes/_index.md (add a row) and log.md (one line).
6. Delete ${markerPath}.`;
}

main().catch(err => {
  log(`[cc-knowledge] Error: ${err.message}`);
  process.exit(0);
});
