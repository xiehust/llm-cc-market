const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { getWikiHub, analyzeSession, MIN_SESSION_SCORE } = require('./lib/utils');

let txCounter = 0;
function writeTranscript(entries) {
  const file = path.join(os.tmpdir(), `cc-knowledge-tx-${process.pid}-${txCounter++}.jsonl`);
  fs.writeFileSync(file, entries.map(e => JSON.stringify(e)).join('\n'), 'utf8');
  return file;
}

test('analyzeSession scores a lone trivial edit below threshold', () => {
  const file = writeTranscript([
    { type: 'user', content: 'fix typo in readme' },
    { type: 'assistant', content: 'ok' },
    { type: 'assistant', content: [{ type: 'tool_use', name: 'Edit', input: { file_path: '/repo/README.md' } }] }
  ]);
  try {
    const s = analyzeSession(file);
    assert.equal(s.editCount, 1);
    assert.equal(s.distinctFiles, 1);
    assert.ok(s.score < MIN_SESSION_SCORE, `expected score ${s.score} < ${MIN_SESSION_SCORE}`);
  } finally {
    fs.rmSync(file, { force: true });
  }
});

test('analyzeSession credits an error→fix sequence', () => {
  const file = writeTranscript([
    { type: 'user', content: 'run the build' },
    { type: 'user', content: [{ type: 'tool_result', is_error: true, content: 'TypeError: boom' }] },
    { type: 'assistant', content: [{ type: 'tool_use', name: 'Edit', input: { file_path: '/repo/src/a.js' } }] }
  ]);
  try {
    const s = analyzeSession(file);
    assert.equal(s.errorCount, 1);
    assert.equal(s.errorFixCount, 1);
    assert.ok(s.score >= MIN_SESSION_SCORE, `expected score ${s.score} >= ${MIN_SESSION_SCORE}`);
  } finally {
    fs.rmSync(file, { force: true });
  }
});

test('analyzeSession credits a user correction', () => {
  const file = writeTranscript([
    { type: 'user', content: 'add the helper' },
    { type: 'user', content: "no, that's wrong — use the async variant instead" },
    { type: 'assistant', content: [{ type: 'tool_use', name: 'Edit', input: { file_path: '/repo/src/b.js' } }] },
    { type: 'assistant', content: [{ type: 'tool_use', name: 'Edit', input: { file_path: '/repo/src/c.js' } }] }
  ]);
  try {
    const s = analyzeSession(file);
    assert.equal(s.correctionCount, 1);
    assert.equal(s.distinctFiles, 2);
    assert.ok(s.score >= MIN_SESSION_SCORE, `expected score ${s.score} >= ${MIN_SESSION_SCORE}`);
  } finally {
    fs.rmSync(file, { force: true });
  }
});

test('analyzeSession returns zero for missing transcript', () => {
  const s = analyzeSession('/nonexistent/path.jsonl');
  assert.equal(s.score, 0);
  assert.equal(s.msgCount, 0);
});

function withTempHome(fn) {
  const previousHome = process.env.HOME;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'cc-knowledge-home-'));
  process.env.HOME = home;
  try {
    return fn(home);
  } finally {
    process.env.HOME = previousHome;
    fs.rmSync(home, { recursive: true, force: true });
  }
}

function writeConfig(home, config) {
  const configDir = path.join(home, '.config', 'llm-wiki');
  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(path.join(configDir, 'config.json'), JSON.stringify(config), 'utf8');
}

test('getWikiHub keeps configured hub_path authoritative when resolved_path is not initialized', () => {
  withTempHome(home => {
    writeConfig(home, {
      hub_path: '~/Library/Mobile Documents/com~apple~CloudDocs/wiki',
      resolved_path: path.join(home, 'old-machine-wiki')
    });
    fs.mkdirSync(path.join(home, 'old-machine-wiki'), { recursive: true });

    assert.equal(
      getWikiHub(),
      path.join(home, 'Library', 'Mobile Documents', 'com~apple~CloudDocs', 'wiki')
    );
  });
});

test('getWikiHub uses initialized resolved_path only when hub_path is absent', () => {
  withTempHome(home => {
    const resolvedPath = path.join(home, 'legacy-wiki');
    writeConfig(home, { resolved_path: resolvedPath });
    fs.mkdirSync(resolvedPath, { recursive: true });
    fs.writeFileSync(path.join(resolvedPath, '_index.md'), '# Legacy wiki\n', 'utf8');

    assert.equal(getWikiHub(), resolvedPath);
  });
});

test('getWikiHub surfaces permission errors for configured hub_path', () => {
  withTempHome(home => {
    writeConfig(home, { hub_path: '~/private-wiki' });

    const originalStatSync = fs.statSync;
    fs.statSync = filePath => {
      if (filePath === path.join(home, 'private-wiki')) {
        const err = new Error('operation not permitted');
        err.code = 'EPERM';
        throw err;
      }
      return originalStatSync(filePath);
    };
    try {
      assert.throws(() => getWikiHub(), /operation not permitted/);
    } finally {
      fs.statSync = originalStatSync;
    }
  });
});
