#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { getPendingDir, readFile, log } = require('./lib/utils');

async function main() {
  const pendingDir = getPendingDir();
  if (!fs.existsSync(pendingDir)) {
    process.exit(0);
  }

  let files;
  try {
    files = fs.readdirSync(pendingDir).filter(f => f.endsWith('.json'));
  } catch {
    process.exit(0);
  }

  if (files.length === 0) {
    process.exit(0);
  }

  const deferred = [];
  const stale = [];
  const completed = [];

  for (const file of files) {
    const markerPath = path.join(pendingDir, file);
    const content = readFile(markerPath);
    if (!content) continue;

    try {
      const marker = JSON.parse(content);
      const ageMs = Date.now() - new Date(marker.timestamp).getTime();
      const ageMin = ageMs / 60000;

      if (marker.status === 'deferred') {
        deferred.push(marker);
      } else if (marker.status === 'spawned' && ageMin > 5) {
        stale.push({ marker, path: markerPath });
      } else if (marker.status === 'completed') {
        completed.push(markerPath);
      }
    } catch {}
  }

  // Clean up completed markers
  for (const p of completed) {
    try { fs.unlinkSync(p); } catch {}
  }

  // Report deferred cultivations
  if (deferred.length > 0) {
    log(`[cc-knowledge] ${deferred.length} session(s) pending cultivation:`);
    for (const m of deferred.slice(0, 3)) {
      log(`  • ${m.date} — "${m.topicHint}" (${m.gateSignals.msgCount} msgs)`);
    }
    log('[cc-knowledge] Run /cc-knowledge:cultivate to process');
  }

  // Warn about stale spawned processes
  if (stale.length > 0) {
    log(`[cc-knowledge] ${stale.length} extraction(s) may have failed — run /cc-knowledge:cultivate --retry`);
    for (const { marker, path: mp } of stale) {
      // Promote to deferred so user can retry
      marker.status = 'deferred';
      try {
        fs.writeFileSync(mp, JSON.stringify(marker, null, 2));
      } catch {}
    }
  }

  process.exit(0);
}

main().catch(() => process.exit(0));
