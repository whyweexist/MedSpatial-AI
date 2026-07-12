/**
 * MedSpatial AI — Chat Panel Component
 * Conversational AI panel for medical Q&A about the current scan.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { sendChatMessage } from '../services/api';

export default function ChatPanel({ scanId, sessionId, onSessionChange, findings }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your MedSpatial AI assistant. Upload a DICOM scan and I can help you analyze it. Ask me about findings, anatomy, tissue layers, or anything about the scan.',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Quick suggestion chips
  const suggestions = [
    'What findings were detected?',
    'Tell me about the tissue layers',
    'What is a pulmonary nodule?',
    'Show scan information',
  ];

  const handleSend = useCallback(async (messageText) => {
    const msg = messageText || input.trim();
    if (!msg || loading) return;

    // Add user message
    const userMessage = { role: 'user', content: msg };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    if (!scanId) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Please upload a DICOM scan first. Once you have an active scan, I can answer questions about it, run analyses, and help you explore the 3D model.',
      }]);
      setLoading(false);
      return;
    }

    try {
      const response = await sendChatMessage(scanId, msg, sessionId);
      onSessionChange(response.session_id);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.response,
        referencedRegions: response.referenced_regions,
      }]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
      }]);
    }

    setLoading(false);
  }, [input, loading, scanId, sessionId, onSessionChange]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-container" id="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ 
            width: 28, height: 28, borderRadius: 6,
            background: 'var(--accent-gradient)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14,
          }}>
            🤖
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Medical AI Assistant</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {scanId ? 'Analyzing active scan' : 'Waiting for scan'}
            </div>
          </div>
        </div>
        {findings && findings.length > 0 && (
          <span className="badge badge-warning">{findings.length} findings</span>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className={`chat-avatar ${msg.role === 'assistant' ? 'ai' : 'user-avatar'}`}>
              {msg.role === 'assistant' ? 'AI' : 'U'}
            </div>
            <div className="chat-bubble">
              {msg.content.split('\n').map((line, j) => (
                <React.Fragment key={j}>
                  {line.startsWith('**') && line.endsWith('**') ? (
                    <strong>{line.replace(/\*\*/g, '')}</strong>
                  ) : line.startsWith('- ') ? (
                    <div style={{ marginLeft: 8, marginTop: 2 }}>• {line.slice(2)}</div>
                  ) : (
                    line
                  )}
                  {j < msg.content.split('\n').length - 1 && <br />}
                </React.Fragment>
              ))}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="chat-avatar ai">AI</div>
            <div className="chat-bubble" style={{ display: 'flex', gap: 4, padding: '12px 16px' }}>
              <span className="spinner" style={{ width: 14, height: 14 }}></span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestion chips */}
      {messages.length <= 2 && !loading && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          padding: '0 16px 12px',
        }}>
          {suggestions.map((suggestion, i) => (
            <button
              key={i}
              className="btn btn-secondary btn-sm"
              onClick={() => handleSend(suggestion)}
              style={{ fontSize: 11 }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the scan..."
            id="chat-input"
          />
          <button
            className="btn btn-primary btn-icon"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            id="chat-send"
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
