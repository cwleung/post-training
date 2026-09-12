import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/shared/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-400',
  {
    variants: {
      variant: {
        default:
          'bg-primary/15 text-primary border border-primary/30',
        secondary:
          'bg-secondary text-secondary-foreground border border-border',
        destructive:
          'bg-destructive/15 text-destructive border border-destructive/30',
        emerald:
          'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30',
        amber:
          'bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30',
        rose:
          'bg-rose-500/15 text-rose-600 dark:text-rose-400 border border-rose-500/30',
        violet:
          'bg-purple-500/15 text-purple-600 dark:text-purple-400 border border-purple-500/30',
        outline:
          'text-muted-foreground border border-border bg-transparent',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
