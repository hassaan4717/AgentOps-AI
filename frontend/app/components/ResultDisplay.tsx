'use client'

// =============================================================================
// ResultDisplay Component
// =============================================================================
// Displays the result of agent execution, including:
// - Final output
// - Evaluation (passed, score, reasons)
// - Memory usage indicator
// - Error messages (if any)

interface Evaluation {
  passed: boolean
  score: number
  reasons: string[]
}

interface ResultDisplayProps {
  result: string | null
  evaluation: Evaluation | null
  memoryUsed: boolean | null
  error: string | null
}

export default function ResultDisplay({
  result,
  evaluation,
  memoryUsed,
  error,
}: ResultDisplayProps) {
  // =============================================================================
  // Error Display
  // =============================================================================

  if (error) {
    return (
      <div className="card" style={{ borderLeft: '4px solid var(--color-error)' }}>
        <h2>Error</h2>
        <p style={{ color: 'var(--color-error)', marginBottom: 'var(--space-lg)' }}>{error}</p>
        <p className="text-muted" style={{ marginBottom: 'var(--space-sm)' }}>
          Common causes:
        </p>
        <ul className="text-muted" style={{ marginLeft: 'var(--space-lg)', lineHeight: 1.8 }}>
          <li>Backend not running</li>
          <li>API key missing or invalid</li>
          <li>Rate limit exceeded</li>
          <li>Network connection issue</li>
        </ul>
      </div>
    )
  }

  // =============================================================================
  // Success Display
  // =============================================================================

  if (!result || !evaluation) {
    return null
  }

  return (
    <div>
      {/* System Feedback Section */}
      <div className="card">
        <h2>Evaluation</h2>
        <p className="text-muted" style={{ marginBottom: 'var(--space-lg)' }}>
          System evaluates outputs before returning results.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 'var(--space-md)',
            marginBottom: 'var(--space-lg)',
          }}
        >
          {/* Evaluation Score */}
          <div
            style={{
              padding: 'var(--space-md)',
              background: 'var(--color-bg)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--color-border)',
              textAlign: 'center',
            }}
          >
            <div className="text-muted" style={{ fontSize: '0.6875rem', marginBottom: 'var(--space-sm)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Score
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              {evaluation.score}
              <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--color-text-secondary)' }}> / 10</span>
            </div>
          </div>

          {/* Pass/Fail Status */}
          <div
            style={{
              padding: 'var(--space-md)',
              background: 'var(--color-bg)',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${evaluation.passed ? 'var(--color-success)' : 'var(--color-error)'}`,
              textAlign: 'center',
            }}
          >
            <div
              className="text-muted"
              style={{
                fontSize: '0.6875rem',
                marginBottom: 'var(--space-sm)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              Status
            </div>
            <div
              style={{
                fontSize: '1.25rem',
                fontWeight: 600,
                color: evaluation.passed ? 'var(--color-success)' : 'var(--color-error)',
              }}
            >
              {evaluation.passed ? 'Pass' : 'Fail'}
            </div>
          </div>

          {/* Memory Used */}
          {memoryUsed !== null && (
            <div
              style={{
                padding: 'var(--space-md)',
                background: 'var(--color-bg)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border)',
                textAlign: 'center',
              }}
            >
              <div className="text-muted" style={{ fontSize: '0.6875rem', marginBottom: 'var(--space-sm)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Memory
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                {memoryUsed ? 'Yes' : 'No'}
              </div>
              <div className="text-muted" style={{ fontSize: '0.6875rem', marginTop: 'var(--space-xs)' }}>
                {memoryUsed ? 'Used' : 'Fresh'}
              </div>
            </div>
          )}
        </div>

        {/* Evaluation Reasons */}
        {evaluation.reasons && evaluation.reasons.length > 0 && (
          <div>
            <strong style={{ display: 'block', marginBottom: 'var(--space-sm)', fontSize: '0.875rem' }}>
              Details
            </strong>
            <ul style={{ marginLeft: 'var(--space-lg)', lineHeight: 1.8 }}>
              {evaluation.reasons.map((reason, index) => (
                <li key={index} className="text-muted" style={{ fontSize: '0.875rem' }}>
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Final Output */}
      <div className="card">
        <h2>Output</h2>
        <div
          style={{
            padding: 'var(--space-md)',
            background: 'var(--color-bg)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border)',
            whiteSpace: 'pre-wrap',
            lineHeight: 1.6,
            fontSize: '0.9375rem',
            maxHeight: '500px',
            overflowY: 'auto',
            color: 'var(--color-text-primary)',
          }}
        >
          {result}
        </div>

        {/* Copy Button */}
        <button
          onClick={() => {
            navigator.clipboard.writeText(result)
            alert('Copied to clipboard')
          }}
          style={{
            marginTop: 'var(--space-md)',
            padding: '0.5rem 1rem',
            background: 'var(--color-bg)',
            color: 'var(--color-text-primary)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.8125rem',
            cursor: 'pointer',
          }}
        >
          Copy
        </button>
      </div>
    </div>
  )
}
