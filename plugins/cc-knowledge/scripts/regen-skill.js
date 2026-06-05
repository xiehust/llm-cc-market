#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { getWikiHub, getHomeDir, readFile, writeFile, ensureDir, getDateString } = require('./lib/utils');

const topicName = process.argv[2];
if (!topicName) {
  console.error('Usage: node regen-skill.js <topic-name>');
  process.exit(1);
}

function main() {
  const hubPath = getWikiHub();
  const topicPath = path.join(hubPath, 'topics', topicName);

  if (!fs.existsSync(topicPath)) {
    console.error(`Topic not found: ${topicPath}`);
    process.exit(1);
  }

  const notesDir = path.join(topicPath, 'raw', 'notes');
  if (!fs.existsSync(notesDir)) {
    console.error(`No notes directory: ${notesDir}`);
    process.exit(1);
  }

  // Glob all .md note files (exclude _index.md)
  // Performance cap: never scan more than MAX_NOTES_TO_SCAN files; archive horizon: skip notes older than ARCHIVE_DAYS
  const MAX_NOTES_TO_SCAN = 100;
  const ARCHIVE_DAYS = 365;
  const cutoffMs = Date.now() - ARCHIVE_DAYS * 24 * 60 * 60 * 1000;

  const allNotes = fs.readdirSync(notesDir)
    .filter(f => f.endsWith('.md') && f !== '_index.md')
    .sort();

  // Drop notes whose filename date is older than ARCHIVE_DAYS (effectively dormant per DDD lifecycle)
  const liveNotes = allNotes.filter(f => {
    const dateMatch = f.match(/^(\d{4}-\d{2}-\d{2})-/);
    if (!dateMatch) return true; // keep if filename has no date
    const fileMs = new Date(dateMatch[1]).getTime();
    return Number.isFinite(fileMs) ? fileMs >= cutoffMs : true;
  });

  // Cap scan size: keep most recent MAX_NOTES_TO_SCAN
  const noteFiles = liveNotes.slice(-MAX_NOTES_TO_SCAN);

  if (noteFiles.length === 0) {
    console.error('No lesson notes found');
    process.exit(0);
  }

  // Per-file age in days, used for time-decay scoring
  const today = Date.now();
  const fileAges = noteFiles.map(f => {
    const dateMatch = f.match(/^(\d{4}-\d{2}-\d{2})-/);
    if (!dateMatch) return 0;
    const fileMs = new Date(dateMatch[1]).getTime();
    if (!Number.isFinite(fileMs)) return 0;
    return Math.max(0, Math.floor((today - fileMs) / (24 * 60 * 60 * 1000)));
  });

  // Extract rules and pitfalls from all notes
  const rules = [];
  const pitfalls = [];
  const allTags = new Set();
  let totalLessons = 0;

  for (let i = 0; i < noteFiles.length; i++) {
    const filePath = path.join(notesDir, noteFiles[i]);
    const content = readFile(filePath);
    if (!content) continue;

    // Count lessons
    const lessonMatches = content.match(/^## Lesson \d+/gm);
    if (lessonMatches) totalLessons += lessonMatches.length;

    // Extract tags from frontmatter
    const tagMatch = content.match(/^tags:\s*\[([^\]]+)\]/m);
    if (tagMatch) {
      tagMatch[1].split(',').map(t => t.trim()).forEach(t => allTags.add(t));
    }

    // Walk lesson sections; capture category + rule together so we can route by judgment dimension
    const sections = content.split(/^## Lesson \d+/m).slice(1);
    for (const section of sections) {
      const catMatch = section.match(/\*\*Category\*\*:\s*([a-zA-Z]+)/);
      const ruleMatch = section.match(/\*\*Rule\*\*:\s*(.+)/);
      const category = catMatch ? catMatch[1].trim().toLowerCase() : '';
      if (ruleMatch) {
        rules.push({
          text: ruleMatch[1].trim(),
          category,
          fileIndex: i,
          file: noteFiles[i]
        });
      }

      if (category === 'gotcha') {
        const symptom = section.match(/\*\*Symptom\*\*:\s*(.+)/);
        const rootCause = section.match(/\*\*Root cause\*\*:\s*(.+)/);
        const fix = section.match(/\*\*Fix\*\*:\s*(.+)/);
        if (symptom && rootCause && fix) {
          pitfalls.push({
            symptom: symptom[1].trim(),
            rootCause: rootCause[1].trim(),
            fix: fix[1].trim(),
            fileIndex: i
          });
        }
      }
    }
  }

  // Map lesson category to DDD-style judgment dimension.
  // tech: pattern/rule/discovery — what to do (TECH visit)
  // improvement: gotcha/correction — what failed before (IMPROVEMENT visit)
  function judgmentDim(category) {
    const c = (category || '').toLowerCase();
    if (c === 'gotcha' || c === 'correction') return 'improvement';
    return 'tech';
  }

  // Deduplicate rules
  const deduped = deduplicateRules(rules);

  // Time-decay factor (DDD lifecycle: dormant after ~30 days, near-archived by ~90).
  // Score = frequency × 2 × decay(age) + decay(age)
  // decay(age) = max(0.1, 1 - ageDays / 180): full weight today, ~83% at 30d, ~50% at 90d, floor 10% past 162d.
  function decay(ageDays) {
    return Math.max(0.1, 1 - ageDays / 180);
  }

  const ranked = deduped
    .map(r => {
      const age = fileAges[r.fileIndex] || 0;
      const d = decay(age);
      return { ...r, score: r.count * 2 * d + d };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  // Top pitfalls by recency
  const topPitfalls = pitfalls
    .sort((a, b) => b.fileIndex - a.fileIndex)
    .slice(0, 5);

  // Count proposal review artifacts in llm-wiki's maintenance area.
  const proposalsDir = path.join(topicPath, '.librarian', 'proposals');
  let proposalCount = 0;
  if (fs.existsSync(proposalsDir)) {
    proposalCount = fs.readdirSync(proposalsDir).filter(f => f.endsWith('.proposal.md')).length;
  }

  // Build skill content
  const displayName = topicName.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const lastDate = noteFiles.length > 0 ? noteFiles[noteFiles.length - 1].slice(0, 10) : getDateString();
  const tagKeywords = [...allTags].filter(t => t !== 'lessons-learned').slice(0, 8).join(', ');
  const top3Rules = ranked.slice(0, 3).map(r => r.text.slice(0, 60)).join('; ');

  const description = `Cultivated lessons on ${topicName}: ${top3Rules}. Invoke when working on ${tagKeywords || topicName}.`;

  let skillContent = `---
name: cc-knowledge-${topicName}
description: "${description.replace(/"/g, '\\"')}"
---

# ${displayName} — Cultivated Knowledge

${totalLessons} lessons across ${noteFiles.length} sessions. Last cultivated: ${lastDate}.

`;

  // Partition by judgment dimension (DDD-style): tech (what to do) vs improvement (what failed before)
  const techRules = ranked.filter(r => judgmentDim(r.category) === 'tech');
  const improvementRules = ranked.filter(r => judgmentDim(r.category) === 'improvement');

  if (techRules.length > 0) {
    skillContent += `## What to Do (TECH)\n\n`;
    techRules.forEach((r, idx) => {
      skillContent += `${idx + 1}. ${r.text}\n`;
    });
    skillContent += `\n`;
  }

  if (improvementRules.length > 0) {
    skillContent += `## What Failed Before (IMPROVEMENT)\n\n`;
    improvementRules.forEach((r, idx) => {
      skillContent += `${idx + 1}. ${r.text}\n`;
    });
    skillContent += `\n`;
  }

  // Fallback: if neither bucket has content (unusual — happens only with no Category fields), list everything
  if (techRules.length === 0 && improvementRules.length === 0 && ranked.length > 0) {
    skillContent += `## Top Rules\n\n`;
    ranked.forEach((r, idx) => {
      skillContent += `${idx + 1}. ${r.text}\n`;
    });
    skillContent += `\n`;
  }

  if (topPitfalls.length > 0) {
    skillContent += `\n## Quick Pitfalls\n\n| Symptom | Root Cause | Fix |\n|---|---|---|\n`;
    for (const p of topPitfalls) {
      skillContent += `| ${p.symptom.slice(0, 60)} | ${p.rootCause.slice(0, 60)} | ${p.fix.slice(0, 60)} |\n`;
    }
  }

  skillContent += `
## Dive Deeper

- [Topic index](~/wiki/topics/${topicName}/_index.md)
- [Recent lessons](~/wiki/topics/${topicName}/raw/notes/)
- [Compiled articles](~/wiki/topics/${topicName}/wiki/concepts/)
`;

  if (proposalCount > 0) {
    skillContent += `\n## Pending Proposals\n\n${proposalCount} proposal(s) awaiting review. Run \`/cc-knowledge:review --wiki ${topicName}\`.\n`;
  }

  // Write skill file
  const skillDir = path.join(getHomeDir(), '.claude', 'skills', `cc-knowledge-${topicName}`);
  const skillPath = path.join(skillDir, 'SKILL.md');
  ensureDir(skillDir);
  writeFile(skillPath, skillContent);

  console.log(`Regenerated: ${skillPath}`);
  console.log(`  Rules: ${ranked.length}, Pitfalls: ${topPitfalls.length}, Total lessons: ${totalLessons}`);
}

function deduplicateRules(rules) {
  const groups = [];
  for (const rule of rules) {
    const normalized = rule.text.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
    let found = false;
    for (const group of groups) {
      const groupNorm = group.normalized;
      if (normalized.includes(groupNorm) || groupNorm.includes(normalized)) {
        group.count++;
        if (rule.fileIndex > group.fileIndex) {
          group.fileIndex = rule.fileIndex;
          group.category = rule.category; // adopt the freshest occurrence's category
        }
        if (rule.text.length > group.text.length) {
          group.text = rule.text;
          group.normalized = normalized;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      groups.push({
        text: rule.text,
        normalized,
        count: 1,
        fileIndex: rule.fileIndex,
        category: rule.category
      });
    }
  }
  return groups;
}

main();
