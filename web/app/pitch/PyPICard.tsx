'use client';

import { useState } from 'react';
import { getAssetPath } from '../../lib/paths';

export default function PyPICard() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText('pip install blaze2d');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      background: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '16px',
      padding: '2rem 2.5rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '1.25rem',
      maxWidth: '400px',
      backdropFilter: 'blur(10px)',
    }}>
      {/* Python Icon */}
      <img
        src={getAssetPath('/icons/python.svg')}
        alt="Python"
        style={{
          width: '64px',
          height: '64px',
        }}
      />

      <div style={{ textAlign: 'center' }}>
        <h3 style={{
          fontSize: '1.25rem',
          fontWeight: 600,
          color: 'white',
          margin: 0,
          marginBottom: '0.5rem',
        }}>
          Available on PyPI
        </h3>
      </div>

      {/* Install Command */}
      <button
        onClick={handleCopy}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '8px',
          padding: '0.75rem 1.25rem',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          width: '100%',
          justifyContent: 'center',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.5)';
          e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.3)';
          e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        }}
      >
        <code style={{
          fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          fontSize: '0.95rem',
          color: '#64c8ff',
        }}>
          pip install blaze2d
        </code>
        <span style={{
          fontSize: '0.8rem',
          color: copied ? '#00ff88' : 'rgba(255, 255, 255, 0.4)',
          minWidth: '40px',
        }}>
          {copied ? '✓' : 'copy'}
        </span>
      </button>

      {/* PyPI Link */}
      <a
        href="https://pypi.org/project/blaze2d/"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          fontSize: '0.85rem',
          color: 'rgba(255, 255, 255, 0.4)',
          textDecoration: 'none',
          transition: 'color 0.2s ease',
        }}
        onMouseEnter={(e) => e.currentTarget.style.color = '#64c8ff'}
        onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255, 255, 255, 0.4)'}
      >
        View on pypi.org →
      </a>
    </div>
  );
}
