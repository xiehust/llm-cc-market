const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { getWikiHub } = require('./lib/utils');

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
