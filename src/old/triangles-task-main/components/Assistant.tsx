import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from "@google/genai";
import { MessageSquare, Send, Sparkles, Loader2 } from 'lucide-react';
import { ChatMessage, LabConfig, TrialResult } from '../types';

interface AssistantProps {
  config: LabConfig;
  history: TrialResult[];
}

const Assistant: React.FC<AssistantProps> = ({ config, history }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'model', text: "Hello! I am your Research Assistant. I can help you understand the Triangles Task, explain Hazard Rates, or analyze your latest performance data. What's on your mind?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      // Initialize Gemini
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Prepare context
      const context = `
        Current Experiment Config:
        - Hazard Rate (H): ${config.hazardRate}
        - Distribution Sigma: ${config.distributionSigma}
        - Total Trials Run: ${history.length}
        - Accuracy: ${history.length > 0 ? (history.filter(h => h.isCorrect).length / history.length * 100).toFixed(1) + '%' : 'N/A'}
        
        The user is participating in a "Triangles Task" (Glaze et al., 2015) experiment which measures belief updating.
        Answer their questions about computational psychiatry, the math behind the task (Bayesian inference), or their performance.
        Be concise, helpful, and scientific but accessible.
      `;

      // Use generateContent with system instruction
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
            ...messages.map(m => ({ 
                role: m.role, 
                parts: [{ text: m.text }] 
            })),
            { role: 'user', parts: [{ text: input }] }
        ],
        config: {
            systemInstruction: context
        }
      });

      const text = response.text || "I couldn't generate a response.";
      
      setMessages(prev => [...prev, { role: 'model', text }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'model', text: "Error connecting to AI Assistant." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-slate-900 rounded-xl overflow-hidden border border-slate-700 shadow-2xl">
      {/* Header */}
      <div className="bg-slate-800 p-4 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
            <div className="bg-indigo-500/20 p-2 rounded-lg">
                <Sparkles className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
                <h3 className="font-bold text-white">Lab Assistant</h3>
                <p className="text-xs text-indigo-300">Powered by Gemini 3 Flash</p>
            </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div 
            key={idx} 
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user' 
                  ? 'bg-indigo-600 text-white rounded-br-none' 
                  : 'bg-slate-800 text-slate-200 rounded-bl-none border border-slate-700'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start">
                <div className="bg-slate-800 rounded-2xl rounded-bl-none px-4 py-3 border border-slate-700">
                    <Loader2 className="w-5 h-5 text-slate-400 animate-spin" />
                </div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-slate-800 border-t border-slate-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about hazard rates, analysis, or the task..."
            className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
          />
          <button 
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white p-3 rounded-lg transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Assistant;