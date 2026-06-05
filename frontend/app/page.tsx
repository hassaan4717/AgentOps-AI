'use client'

import { useState, useEffect } from 'react'
import AgentInput from './components/AgentInput'
import ResultDisplay from './components/ResultDisplay'
import HistoryList from './components/HistoryList'

// =============================================================================
// Types
// =============================================================================

interface Evaluation {
  passed: boolean
  score: number
  reasons: string[]
}

interface ApiResponse {
  final_output: string
  evaluation: Evaluation
  memory_used: boolean
}

interface ErrorResponse {
  error: string
  message: string
  details?: string
}

interface MemoryRecord {
  user_goal: string
  summary: string
  score: number
  created_at: string
}

interface HistoryResponse {
  memories: MemoryRecord[]
  total: number
}

const API_URL = 'http://localhost:8000'

export default function Home() {
  // =============================================================================
  // State Management
  // =============================================================================

  const [goal, setGoal] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [memoryUsed, setMemoryUsed] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<MemoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)

  // =============================================================================
  // Load History on Mount
  // =============================================================================
  // Fetch past successful tasks once when the page loads
  // No auto-refresh - keeps network usage minimal

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`${API_URL}/history`)
        if (response.ok) {
          const data: HistoryResponse = await response.json()
          setHistory(data.memories)
        } else {
          // History endpoint failed - not critical, just log it
          console.warn('Failed to load history')
        }
      } catch (err) {
        // Network error loading history - not critical
        console.warn('Could not fetch history:', err)
      } finally {
        setHistoryLoading(false)
      }
    }

    fetchHistory()
  }, []) // Empty dependency array = run once on mount

  // =============================================================================
  // API Call Handler
  // =============================================================================

  const handleRunAgent = async (goalText: string) => {
    setError(null)
    setResult(null)
    setEvaluation(null)
    setMemoryUsed(null)
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ goal: goalText }),
      })

      const data = await response.json()

      if (!response.ok) {
        const errorData = data.detail as ErrorResponse
        throw new Error(errorData?.message || 'Failed to execute task')
      }

      const apiResponse = data as ApiResponse
      setResult(apiResponse.final_output)
      setEvaluation(apiResponse.evaluation)
      setMemoryUsed(apiResponse.memory_used)

      // Refresh history if task passed with high score
      if (apiResponse.evaluation.passed && apiResponse.evaluation.score >= 8) {
        setTimeout(async () => {
          try {
            const historyResponse = await fetch(`${API_URL}/history`)
            if (historyResponse.ok) {
              const historyData: HistoryResponse = await historyResponse.json()
              setHistory(historyData.memories)
            }
          } catch (err) {
            console.warn('Failed to refresh history:', err)
          }
        }, 1000)
      }
      
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('An unexpected error occurred. Please try again.')
      }
      console.error('Agent execution error:', err)
      
    } finally {
      setLoading(false)
    }
  }

  // =============================================================================
  // Clear Results Handler
  // =============================================================================

  const handleClear = () => {
    setGoal('')
    setResult(null)
    setEvaluation(null)
    setMemoryUsed(null)
    setError(null)
  }

  return (
    <main className="container">
      {/* Header Section */}
      <header className="header">
        <div className="badge">Beta</div>
        <h1>AgentOps AI</h1>
        <p>
          Multi-agent task execution system
        </p>
      </header>

      {/* Main Card - Input Form */}
      <div className="card">
        <h2>Task Input</h2>
        <p className="text-muted" style={{ marginBottom: 'var(--space-lg)' }}>
          Enter a goal. The system plans, executes, and evaluates the output.
        </p>

        <AgentInput
          goal={goal}
          setGoal={setGoal}
          loading={loading}
          onSubmit={handleRunAgent}
          onClear={handleClear}
        />
      </div>

      {/* Result Display */}
      {(result || error) && (
        <ResultDisplay
          result={result}
          evaluation={evaluation}
          memoryUsed={memoryUsed}
          error={error}
        />
      )}

      {/* Info Section - Only show when no results */}
      {!result && !error && (
        <div className="card">
          <h2>Agent Pipeline</h2>
          <div style={{ display: 'grid', gap: 'var(--space-lg)' }}>
            <div>
              <strong>Supervisor</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Plans steps and determines if research is required.
              </p>
            </div>
            <div>
              <strong>Research (Conditional)</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Gathers context and information when needed.
              </p>
            </div>
            <div>
              <strong>Execution</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Generates output based on plan and research.
              </p>
            </div>
            <div>
              <strong>Evaluation</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Validates output. Retries up to 5 times if needed.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Past Successful Tasks - Always visible */}
      <HistoryList history={history} loading={historyLoading} />

      {/* Footer */}
      <footer style={{ textAlign: 'center', color: 'var(--color-text-secondary)', marginTop: 'var(--space-2xl)', marginBottom: 'var(--space-xl)' }}>
        <p style={{ fontSize: '0.8125rem', margin: 0 }}>
          LangGraph • FastAPI • Next.js
        </p>
      </footer>
    </main>
  )
}
