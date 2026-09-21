'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { copyText } from './download';

/**
 * Copy-to-clipboard with a short "copied" acknowledgement.
 *
 * The timer is cleared on unmount so a component that disappears mid-flash does
 * not set state after it is gone.
 */
export function useCopy(resetAfter = 2000) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  const copy = useCallback(
    async (text: string) => {
      const ok = await copyText(text);
      setCopied(ok);
      clearTimeout(timer.current);
      if (ok) timer.current = setTimeout(() => setCopied(false), resetAfter);
      return ok;
    },
    [resetAfter],
  );

  return { copied, copy };
}
