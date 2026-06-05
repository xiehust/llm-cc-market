import type { ReactNode } from 'react';

type BadgeTone = 'amber' | 'blue' | 'green' | 'red' | 'slate' | 'violet';

interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
}

export default function Badge({ children, tone = 'slate' }: BadgeProps) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
