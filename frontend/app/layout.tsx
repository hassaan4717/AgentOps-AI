import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AgentOps AI Platform',
  description: 'Multi-agent AI system powered by LangGraph',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
