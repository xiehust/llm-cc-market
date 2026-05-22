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
  const noteFiles = fs.readdirSync(notesDir)
    .filter(f => f.endsWith('.md') && f !== '_index.md')
    .sort();

  if (noteFiles.length === 0) {
    console.error('No lesson notes found');
    process.exit(0);
  }

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

    // Extract rules
    const ruleMatches = content.matchAll(/^\*\*Rule\*\*:\s*(.+)$/gm);
    for (const m of ruleMatches) {
      rules.push({ text: m[1].trim(), fileIndex: i, file: noteFiles[i] });
    }

    // Extract pitfalls (gotcha category entries)
    const sections = content.split(/^## Lesson \d+/m).slice(1);
    for (const section of sections) {
      if (/\*\*Category\*\*:\s*gotcha/i.test(section)) {
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

  // Deduplicate rules
  const deduped = deduplicateRules(rules);

  // Rank: frequency × 2 + recency
  const ranked = deduped
    .map(r => ({
      ...r,
      score: r.count * 2 + (r.fileIndex / noteFiles.length)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  // Top pitfalls by recency
  const topPitfalls = pitfalls
    .sort((a, b) => b.fileIndex - a.fileIndex)
    .slice(0, 5);

  // Count proposals
  const proposalsDir = path.join(topicPath, 'proposals');
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

## Top Rules

`;

  for (let i = 0; i < ranked.length; i++) {
    skillContent += `${i + 1}. ${ranked[i].text}\n`;
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
        group.fileIndex = Math.max(group.fileIndex, rule.fileIndex);
        if (rule.text.length > group.text.length) {
          group.text = rule.text;
          group.normalized = normalized;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      groups.push({ text: rule.text, normalized, count: 1, fileIndex: rule.fileIndex });
    }
  }
  return groups;
}

main();
