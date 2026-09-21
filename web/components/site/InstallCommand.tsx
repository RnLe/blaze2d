'use client';
import { useCopy } from '@/lib/util/useCopy';

export const INSTALL_COMMAND = 'pip install blaze2d';

/**
 * The `pip install` line, as a button that copies itself.
 *
 * Shared by the documentation footer and the pitch page so the published
 * package name lives in exactly one place.
 */
export function InstallCommand({ size = 'small' }: { size?: 'small' | 'large' }) {
  const { copied, copy } = useCopy();
  return (
    <button
      type="button"
      className={`install-command install-command-${size}`}
      onClick={() => void copy(INSTALL_COMMAND)}
      aria-label={`Copy "${INSTALL_COMMAND}" to the clipboard`}
    >
      <code>{INSTALL_COMMAND}</code>
      <span aria-live="polite">{copied ? '✓' : 'copy'}</span>
    </button>
  );
}
