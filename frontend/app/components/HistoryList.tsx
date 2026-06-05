'use client'

// =============================================================================
// HistoryList Component
// =============================================================================
// Displays past successful tasks from memory storage
// Shows only evaluator-approved results (score >= 8)
// Loads once on page load, no auto-refresh

interface MemoryRecord {
  user_goal: string
  summary: string
  score: number
  created_at: string
}

interface HistoryListProps {
  history: MemoryRecord[]
  loading: boolean
}

export default function HistoryList({ history, loading }: HistoryListProps) {
  // =============================================================================
  // Format Date
  // =============================================================================
  // Convert ISO timestamp to readable format

  const formatDate = (isoString: string): string => {
    try {
      const date = new Date(isoString)
      const now = new Date()
      const diffMs = now.getTime() - date.getTime()
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

      // Show relative time for recent items
      if (diffDays === 0) {
        return 'Today'
      } else if (diffDays === 1) {
        return 'Yesterday'
      } else if (diffDays < 7) {
        return `${diffDays} days ago`
      } else {
        // Show date for older items
        return date.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
        })
      }
    } catch {
      return 'Unknown date'
    }
  }

  // =============================================================================
  // Loading State
  // =============================================================================

  if (loading) {
    return (
      <div className="card">
        <h2>History</h2>
        <p className="text-muted" style={{ marginBottom: 0 }}>
          Loading...
        </p>
      </div>
    )
  }

  // =============================================================================
  // Empty State
  // =============================================================================

  if (history.length === 0) {
    return (
      <div className="card">
        <h2>History</h2>
        <p className="text-muted" style={{ marginBottom: 'var(--space-md)' }}>
          Evaluator-approved tasks only.
        </p>
        <div
          style={{
            padding: 'var(--space-xl)',
            textAlign: 'center',
            background: 'var(--color-bg)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border)',
          }}
        >
          <p className="text-muted" style={{ margin: 0 }}>
            No tasks yet.
          </p>
        </div>
      </div>
    )
  }

  // =============================================================================
  // History Display
  // =============================================================================

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-sm)' }}>
        <h2 style={{ margin: 0 }}>History</h2>
        <span
          style={{
            background: 'var(--color-bg)',
            color: 'var(--color-text-secondary)',
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
            fontSize: '0.6875rem',
            fontWeight: 600,
            border: '1px solid var(--color-border)',
          }}
        >
          {history.length}
        </span>
      </div>
      <p className="text-muted" style={{ marginBottom: 'var(--space-lg)' }}>
        Evaluator-approved tasks only.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {history.map((record, index) => (
          <div
            key={index}
            style={{
              padding: 'var(--space-md)',
              background: 'var(--color-bg)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--color-border)',
              transition: 'border-color 0.15s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-border-hover)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--color-border)'
            }}
          >
            {/* Header Row */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                marginBottom: 'var(--space-sm)',
                gap: 'var(--space-md)',
              }}
            >
              <div style={{ flex: 1 }}>
                <strong
                  style={{
                    display: 'block',
                    fontSize: '0.9375rem',
                    color: 'var(--color-text-primary)',
                    fontWeight: 600,
                  }}
                >
                  {record.user_goal}
                </strong>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-sm)',
                  flexShrink: 0,
                }}
              >
                {/* Score Badge */}
                <div
                  style={{
                    background: 'var(--color-bg-elevated)',
                    color: 'var(--color-text-primary)',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.6875rem',
                    fontWeight: 600,
                    border: '1px solid var(--color-border)',
                  }}
                >
                  {record.score}/10
                </div>
                {/* Date */}
                <span
                  className="text-muted"
                  style={{
                    fontSize: '0.6875rem',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {formatDate(record.created_at)}
                </span>
              </div>
            </div>

            {/* Summary */}
            <p
              className="text-muted"
              style={{
                fontSize: '0.875rem',
                lineHeight: 1.6,
                margin: 0,
              }}
            >
              {record.summary}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
