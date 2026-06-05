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
  const correctionPatterns = /\b(no[,.]?\s+(not|don't|that's wrong)|wrong\s|not that|use .+ instead|actually[,.]?\s+(you should|it should|we should|let's))/i;
  for (const line of lines) {
    if (/"type"\s*:\s*"user"/.test(line) && correctionPatterns.test(line)) {
      return true;
    }
  }
  return false;
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
  readWikisJson,
  getPendingDir,
  log
};
