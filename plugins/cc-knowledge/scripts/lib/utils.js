const fs = require('fs');
const path = require('path');
const os = require('os');

function getHomeDir() {
  return os.homedir();
}

function getWikiHub() {
  const configPath = path.join(getHomeDir(), '.config', 'llm-wiki', 'config.json');
  const content = readFile(configPath);
  if (content) {
    let config;
    try {
      config = JSON.parse(content);
    } catch {}

    if (config) {
      if (config.hub_path) {
        const resolved = expandLeadingTilde(config.hub_path);
        if (pathExists(resolved)) return resolved;
        if (config.resolved_path && isInitializedHub(config.resolved_path)) {
          return config.resolved_path;
        }
        return resolved;
      }
      if (config.resolved_path && isInitializedHub(config.resolved_path)) {
        return config.resolved_path;
      }
    }
  }
  const fallback = path.join(getHomeDir(), 'wiki');
  return fallback;
}

function expandLeadingTilde(value) {
  return value.replace(/^~(?=$|[\\/])/, getHomeDir());
}

function pathExists(filePath) {
  try {
    fs.statSync(filePath);
    return true;
  } catch (err) {
    if (err && (err.code === 'EACCES' || err.code === 'EPERM')) {
      throw err;
    }
    return false;
  }
}

function isInitializedHub(hubPath) {
  return pathExists(path.join(hubPath, '_index.md'));
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
  return dirPath;
}

function readFile(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch {
    return null;
  }
}

function writeFile(filePath, content) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, 'utf8');
}

function appendFile(filePath, content) {
  ensureDir(path.dirname(filePath));
  fs.appendFileSync(filePath, content, 'utf8');
}

function getDateString() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9一-鿿]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40);
}

function countTranscriptMessages(transcriptPath) {
  const content = readFile(transcriptPath);
  if (!content) return 0;
  const matches = content.match(/"type"\s*:\s*"user"/g);
  return matches ? matches.length : 0;
}

function hasEditOrWriteTools(transcriptPath) {
  const content = readFile(transcriptPath);
  if (!content) return false;
  return /"name"\s*:\s*"(Edit|Write)"/.test(content);
}

function hasBashErrors(transcriptPath) {
  const content = readFile(transcriptPath);
  if (!content) return false;
  return /"is_error"\s*:\s*true/.test(content);
}

function hasUserCorrections(transcriptPath) {
  const content = readFile(transcriptPath);
  if (!content) return false;
  const lines = content.split('\n');
  for (const line of lines) {
    if (/"type"\s*:\s*"user"/.test(line) && CORRECTION_PATTERN.test(line)) {
      return true;
    }
  }
  return false;
}

const CORRECTION_PATTERN = /\b(no[,.]?\s+(not|don't|that's wrong)|wrong\s|not that|use .+ instead|actually[,.]?\s+(you should|it should|we should|let's))/i;

const EDIT_TOOL_PATTERN = /"name"\s*:\s*"(Edit|Write|MultiEdit|NotebookEdit)"/g;
const FILE_PATH_PATTERN = /"file_path"\s*:\s*"((?:[^"\\]|\\.)*)"/g;

// Weights tuned for a STRICT gate ("宁可漏，保持 wiki 干净"):
// a lone trivial edit scores 1 and is rejected; real signal (error→fix,
// user corrections, multi-file work) is needed to clear MIN_SESSION_SCORE.
const SCORE_WEIGHTS = {
  errorFix: 3,      // an error followed later by an edit = genuine debugging
  correction: 2,    // a user correcting Claude = a likely durable lesson
  distinctFile: 1,  // breadth of change (capped)
  manyEdits: 1      // bonus when edit activity is clearly non-trivial
};
const DISTINCT_FILE_CAP = 5;
const MANY_EDITS_THRESHOLD = 3;
const MIN_SESSION_SCORE = 4;

/**
 * Single-pass scan of a transcript producing weighted signals used to decide
 * whether a session is worth mining for durable lessons. Returns a `score`
 * plus the raw signal breakdown (also handy for the marker / triage prompt).
 */
function analyzeSession(transcriptPath) {
  const empty = {
    msgCount: 0,
    editCount: 0,
    distinctFiles: 0,
    errorCount: 0,
    errorFixCount: 0,
    correctionCount: 0,
    score: 0
  };
  const content = readFile(transcriptPath);
  if (!content) return empty;

  const lines = content.split('\n');
  const files = new Set();
  let msgCount = 0;
  let editCount = 0;
  let errorCount = 0;
  let errorFixCount = 0;
  let correctionCount = 0;
  let pendingError = false;

  for (const line of lines) {
    if (!line) continue;
    const isUser = /"type"\s*:\s*"user"/.test(line);
    if (isUser) {
      msgCount++;
      if (CORRECTION_PATTERN.test(line)) correctionCount++;
    }

    const edits = line.match(EDIT_TOOL_PATTERN);
    if (edits) {
      editCount += edits.length;
      let m;
      const fileRe = new RegExp(FILE_PATH_PATTERN.source, 'g');
      while ((m = fileRe.exec(line)) !== null) files.add(m[1]);
      // An edit that resolves a previously-seen error = an error→fix sequence.
      if (pendingError) {
        errorFixCount++;
        pendingError = false;
      }
    }

    if (/"is_error"\s*:\s*true/.test(line)) {
      errorCount++;
      pendingError = true;
    }
  }

  const distinctFiles = files.size;
  const score =
    errorFixCount * SCORE_WEIGHTS.errorFix +
    correctionCount * SCORE_WEIGHTS.correction +
    Math.min(distinctFiles, DISTINCT_FILE_CAP) * SCORE_WEIGHTS.distinctFile +
    (editCount > MANY_EDITS_THRESHOLD ? SCORE_WEIGHTS.manyEdits : 0);

  return {
    msgCount,
    editCount,
    distinctFiles,
    errorCount,
    errorFixCount,
    correctionCount,
    score
  };
}

function readWikisJson(hubPath) {
  const wikisPath = path.join(hubPath, 'wikis.json');
  const content = readFile(wikisPath);
  if (!content) return null;
  try {
    return JSON.parse(content);
  } catch {
    return null;
  }
}

function getPendingDir() {
  return path.join(getHomeDir(), '.claude', 'cc-knowledge-pending');
}

function log(msg) {
  console.error(msg);
}

module.exports = {
  getHomeDir,
  getWikiHub,
  ensureDir,
  readFile,
  writeFile,
  appendFile,
  getDateString,
  slugify,
  countTranscriptMessages,
  hasEditOrWriteTools,
  hasBashErrors,
  hasUserCorrections,
  analyzeSession,
  MIN_SESSION_SCORE,
  readWikisJson,
  getPendingDir,
  log
};
