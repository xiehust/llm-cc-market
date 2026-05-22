#!/usr/bin/env node

const path = require('path');
const fs = require('fs');
const { spawn, spawnSync } = require('child_process');
const {
  countTranscriptMessages,
  hasEditOrWriteTools,
  hasBashErrors,
  hasUserCorrections,
  getWikiHub,
  ensureDir,
  writeFile,
  getPendingDir,
  getDateString,
  slugify,
  log
} = require('./lib/utils');

const MIN_MESSAGES = 8;

async function main() {
  const transcriptPath = process.env.CLAUDE_TRANSCRIPT_PATH;
  const sessionId = process.env.CLAUDE_SESSION_ID || 'unknown';

  if (!transcriptPath || !fs.existsSync(transcriptPath)) {
    process.exit(0);
  }

  // Heuristic gate
  const msgCount = countTranscriptMessages(transcriptPath);
  if (msgCount < MIN_MESSAGES) {
    process.exit(0);
  }

  const hasEdits = hasEditOrWriteTools(transcriptPath);
  const hasErrors = hasBashErrors(transcriptPath);
  const hasCorrections = hasUserCorrections(transcriptPath);

  if (!hasEdits && !hasErrors && !hasCorrections) {
    process.exit(0);
  }

  // Gate passed — extract topic hint from transcript
  const topicHint = extractTopicHint(transcriptPath);
  const hubPath = getWikiHub();

  log(`[cc-knowledge] Session qualifies for cultivation (${msgCount} msgs, edits=${hasEdits}, errors=${hasErrors}, corrections=${hasCorrections})`);
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
    gateSignals: { msgCount, hasEdits, hasErrors, hasCorrections },
    status: 'pending'
  };

  const markerPath = path.join(pendingDir, `${sessionId.slice(-12)}.json`);

  // Try to spawn claude for automatic extraction
  const claudePath = findClaude();
  if (claudePath) {
    const skillPath = path.join(__dirname, '..', 'skills', 'cultivator-engine', 'SKILL.md');
    if (fs.existsSync(skillPath)) {
      marker.status = 'spawned';
      writeFile(markerPath, JSON.stringify(marker, null, 2));

      const prompt = buildExtractionPrompt(transcriptPath, topicHint, hubPath, markerPath);
      const child = spawn(claudePath, ['-p', '--model', 'claude-haiku-4-5', prompt], {
        detached: true,
        stdio: 'ignore',
        env: { ...process.env, CC_KNOWLEDGE_MARKER: markerPath }
      });
      child.unref();

      log(`[cc-knowledge] Spawned cultivator (pid ${child.pid})`);
      process.exit(0);
    }
  }

  // Fallback: deferred cultivation
  marker.status = 'deferred';
  writeFile(markerPath, JSON.stringify(marker, null, 2));
  log('[cc-knowledge] Cultivation deferred — run /cc-knowledge:cultivate next session');
  process.exit(0);
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

function buildExtractionPrompt(transcriptPath, topicHint, hubPath, markerPath) {
  const excerptLines = [];
  try {
    const content = fs.readFileSync(transcriptPath, 'utf8');
    const lines = content.split('\n').filter(Boolean);
    const recent = lines.slice(-60);
    for (const line of recent) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'user' || entry.type === 'assistant') {
          const text = typeof entry.content === 'string'
            ? entry.content
            : (Array.isArray(entry.content) ? entry.content.filter(b => b.type === 'text').map(b => b.text).join('\n') : '');
          if (text.length > 0) {
            excerptLines.push(`[${entry.type}]: ${text.slice(0, 500)}`);
          }
        }
      } catch {}
    }
  } catch {}

  const excerpt = excerptLines.slice(-30).join('\n\n');
  return `Extract lessons from this Claude Code session and write them to the wiki at ${hubPath}.
Topic hint: "${topicHint}"
After writing, delete the marker file at: ${markerPath}

Session transcript excerpt:
---
${excerpt}
---

Follow the lesson extraction process:
1. Find error→fix patterns, user corrections, discoveries, gotchas
2. Structure each as: Category (gotcha|pattern|rule|discovery|correction), Context, Symptom, Root cause, Fix, Rule
3. Write to ${hubPath}/topics/<appropriate-topic>/raw/notes/${getDateString()}-ll-${slugify(topicHint)}.md
4. Use llm-wiki frontmatter format (type: lessons-learned, source: session, tags, confidence: high)
5. Target 2-7 lessons. Be specific with error messages and file paths.`;
}

main().catch(err => {
  log(`[cc-knowledge] Error: ${err.message}`);
  process.exit(0);
});
