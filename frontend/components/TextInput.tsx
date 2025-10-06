'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { FeedbackItem } from '@/types'
import { FileText } from 'lucide-react'

interface TextInputProps {
  onDataChange: (data: FeedbackItem[]) => void
}

export default function TextInput({ onDataChange }: TextInputProps) {
  const [text, setText] = useState('')

  const handleTextChange = (value: string) => {
    setText(value)

    if (!value.trim()) {
      onDataChange([])
      return
    }

    // Check if this looks like CSV data (has commas and quotes)
    const isCSV = value.includes(',') && value.includes('"')
    
    let feedbackItems: FeedbackItem[] = []
    
    if (isCSV) {
      // Parse CSV format: ID,"Text","Source"
      const lines = value
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
      
      feedbackItems = lines
        .filter(line => line.includes(','))
        .map((line, index) => {
          // Simple CSV parsing - split by comma and clean up quotes
          const parts = line.split(',').map(part => part.trim().replace(/^"|"$/g, ''))
          return {
            id: parts[0] || (index + 1).toString(),
            text: parts[1] || line,
            source: parts[2] || 'Manual Input',
          }
        })
    } else {
      // Parse as plain text with support for numbered/bulleted lists
      // Pattern: Detect lines that start with numbers (e.g., "12. (Feature Request)")
      const lines = value.split('\n').map(line => line.trim())
      
      // Check if input uses numbered format like "12. (Source)"
      const numberedPattern = /^\d+\.\s*\(([^)]+)\)/
      
      let currentItem: { id: string; text: string; source: string } | null = null
      
      for (const line of lines) {
        if (!line) continue
        
        const match = line.match(numberedPattern)
        
        if (match) {
          // This is a header line like "12. (Feature Request)"
          // Save previous item if exists
          if (currentItem && currentItem.text.trim()) {
            feedbackItems.push(currentItem)
          }
          
          // Start new item
          const source = match[1] || 'Manual Input'
          currentItem = {
            id: feedbackItems.length + 1 + '',
            text: '',
            source: source,
          }
        } else if (currentItem) {
          // This is content for the current item
          currentItem.text += (currentItem.text ? ' ' : '') + line
        } else {
          // No current item and no header - treat as standalone feedback
          // Remove bullet points and list markers
          const cleanLine = line.replace(/^[•\-\*]\s*/, '')
          if (cleanLine && !['•', '-', '*'].includes(cleanLine)) {
            feedbackItems.push({
              id: feedbackItems.length + 1 + '',
              text: cleanLine,
              source: 'Manual Input',
            })
          }
        }
      }
      
      // Don't forget the last item
      if (currentItem && currentItem.text.trim()) {
        feedbackItems.push(currentItem)
      }
    }

    onDataChange(feedbackItems)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="space-y-3 h-full"
    >
      <label className="text-sm font-light text-gray-700 flex items-center gap-2">
        <FileText className="w-4 h-4" strokeWidth={1.5} />
        Enter customer feedback (text or CSV format)
      </label>
      <textarea
        value={text}
        onChange={e => handleTextChange(e.target.value)}
        placeholder="Numbered format with source:&#10;12. (Feature Request)&#10;Need custom filters and scheduled exports.&#10;&#10;13. (App Store)&#10;Love the new features! Great work.&#10;&#10;Or simple list:&#10;The app crashes when I export data&#10;Love the new features!"
        className="w-full h-64 px-4 py-3 rounded-lg border border-gray-200 bg-white focus:ring-1 focus:ring-black focus:border-black transition-all outline-none resize-none text-sm font-light placeholder:text-gray-400"
      />
      <p className="text-xs text-gray-500 font-light">
        💡 Tip: You can start lines with •, -, or * for better formatting
      </p>
    </motion.div>
  )
}
