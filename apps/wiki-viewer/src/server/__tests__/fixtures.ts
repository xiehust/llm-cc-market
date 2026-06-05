import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export async function createFixtureWiki(root: string): Promise<string> {
  const hub = join(root, 'wiki');
  await mkdir(join(hub, 'topics', 'ml-training', 'raw', 'notes'), { recursive: true });
  await mkdir(join(hub, 'topics', 'ml-training', 'wiki', 'concepts'), { recursive: true });
  await mkdir(join(hub, 'topics', 'ml-training', '.librarian', 'proposals'), { recursive: true });
  await mkdir(join(hub, 'topics', '.archive', 'old-topic', 'wiki', 'topics'), { recursive: true });

  await writeFile(
    join(hub, 'wikis.json'),
    JSON.stringify({
      default: '~/wiki',
      wikis: {
        hub: { path: '~/wiki', description: 'Hub' },
        'ml-training': { path: 'topics/ml-training', description: 'Training lessons', status: 'active' },
        'old-topic': { path: 'topics/.archive/old-topic', description: 'Old lessons', status: 'archived' },
      },
      local_wikis: [],
    }),
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', 'raw', 'notes', '2026-06-05-ll-cuda.md'),
    `---
title: "Lessons Learned: CUDA setup"
source: "session"
type: notes
ingested: 2026-06-05
tags: [lessons-learned, cuda, training]
lesson_count: 2
confidence: high
summary: "CUDA package setup"
---
# Lessons Learned: CUDA setup

## Lesson 1: Install keyring first

**Category**: gotcha
**Context**: Installing NVIDIA packages
**Fix**: Install cuda-keyring first
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', 'wiki', 'concepts', 'cuda-packages.md'),
    `---
title: "CUDA Packages"
category: concept
sources: [raw/notes/2026-06-05-ll-cuda.md]
created: 2026-06-05
updated: 2026-06-05
tags: [cuda]
confidence: high
summary: "How CUDA packages fit together"
---
# CUDA Packages

Use the keyring before NVIDIA apt repositories.
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', '.librarian', 'proposals', '2026-06-05-cuda.proposal.md'),
    `---
type: article-append
target: wiki/concepts/cuda-packages.md
date: 2026-06-05
source_lesson: raw/notes/2026-06-05-ll-cuda.md#lesson-1
---
**Proposed append:**

Mention cuda-keyring.
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', '.archive', 'old-topic', 'wiki', 'topics', 'legacy.md'),
    `---
title: "Legacy Topic"
category: topic
tags: [archive]
summary: "Old archived knowledge"
---
# Legacy Topic
Archived material.
`,
    'utf8',
  );

  return hub;
}
