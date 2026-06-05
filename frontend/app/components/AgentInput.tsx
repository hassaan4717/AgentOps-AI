'use client'

// =============================================================================
// AgentInput Component
// =============================================================================
// Controlled form component for entering and submitting agent goals
// State is managed by parent (page.tsx) for coordination with result display

interface AgentInputProps {
  goal: string
  setGoal: (goal: string) => void
  loading: boolean
  onSubmit: (goal: string) => void
  onClear: () => void
}

export default function AgentInput({
  goal,
  setGoal,
  loading,
  onSubmit,
  onClear,
}: AgentInputProps) {
  // =============================================================================
  // Form Submission Handler
  // =============================================================================

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Client-side validation (matches backend requirements)
    if (!goal.trim()) {
      alert('Please enter a goal')
      return
    }

    if (goal.trim().length < 3) {
      alert('Goal must be at least 3 characters')
      return
    }

    if (goal.trim().length > 500) {
      alert('Goal must be less than 500 characters')
      return
    }

    // Call parent handler (which makes API call)
    onSubmit(goal.trim())
  }

  // =============================================================================
  // Clear Handler
  // =============================================================================

  const handleClear = () => {
    onClear()
  }

  // =============================================================================
  // Example Goals
  // =============================================================================

  const exampleGoals = [
    'Explain the benefits of vector databases for semantic search',
    'Compare pros and cons of LangChain vs LangGraph',
    'Write a technical overview of observability in AI systems',
  ]

  const handleExampleClick = (example: string) => {
    setGoal(example)
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="goal">
          Goal
        </label>
        <textarea
          id="goal"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Explain the benefits of vector databases"
          disabled={loading}
          maxLength={500}
        />
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '0.5rem',
          }}
        >
          <span className="text-muted">
            {goal.length}/500 characters
          </span>
          {goal.length >= 3 && goal.length <= 500 && (
            <span className="text-muted" style={{ color: '#10b981' }}>
              ✓ Valid length
            </span>
          )}
        </div>
      </div>

      {/* Example Goals */}
      {goal.length === 0 && (
        <div className="form-group">
          <label>Examples</label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {exampleGoals.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleExampleClick(example)}
                style={{
                  padding: '0.75rem 1rem',
                  background: 'var(--color-bg)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-sm)',
                  textAlign: 'left',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                  color: 'var(--color-text-primary)',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--color-border-hover)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--color-border)'
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <button
          type="submit"
          className="primary"
          disabled={loading || goal.trim().length < 3}
          style={{ flex: 1 }}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Executing...
            </>
          ) : (
            'Execute'
          )}
        </button>

        {goal.length > 0 && !loading && (
          <button
            type="button"
            onClick={handleClear}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'var(--color-bg)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            Clear
          </button>
        )}
      </div>
    </form>
  )
}
