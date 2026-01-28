import React, { useState } from 'react';
import { Play, Pause, RotateCcw, Brain, Lightbulb, Network, RotateCw, Eye, TreePine, Download, FileText } from 'lucide-react';

const ExpansiveInquirySystem = () => {
  const [topic, setTopic] = useState('');
  const [currentStage, setCurrentStage] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState({});
  const [collaborativeMode, setCollaborativeMode] = useState(true);

  const stages = [
    {
      name: "Scope Clarification",
      icon: <Eye className="w-5 h-5" />,
      role: "Scope AI",
      color: "bg-blue-500",
      description: "Refines and clarifies the core inquiry"
    },
    {
      name: "Logical Branching",
      icon: <Brain className="w-5 h-5" />,
      role: "Logic AI",
      color: "bg-green-500",
      description: "Systematic rational exploration"
    },
    {
      name: "Intuitive Branching",
      icon: <Lightbulb className="w-5 h-5" />,
      role: "Mythos AI",
      color: "bg-purple-500",
      description: "Metaphorical and mythopoetic exploration"
    },
    {
      name: "Lateral Exploration",
      icon: <Network className="w-5 h-5" />,
      role: "Bridge AI",
      color: "bg-orange-500",
      description: "Cross-domain connections"
    },
    {
      name: "Recursive Design",
      icon: <RotateCw className="w-5 h-5" />,
      role: "Meta AI",
      color: "bg-red-500",
      description: "Self-improving feedback loops"
    },
    {
      name: "Pattern Recognition",
      icon: <TreePine className="w-5 h-5" />,
      role: "Pattern AI",
      color: "bg-indigo-500",
      description: "Emergent meta-patterns"
    }
  ];

  const generateMarkdownFile = (stageIndex, result) => {
    const stage = stages[stageIndex];
    const timestamp = new Date().toISOString();
    const dateFormatted = new Date().toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    
    const slugTitle = topic.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const stageSlug = stage.name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    
    return `---
title: "${stage.name} - ${topic}"
description: "${stage.description}"
topic: "${topic}"
stage: "${stage.name}"
ai_role: "${stage.role}"
stage_number: ${parseInt(stageIndex) + 1}
total_stages: ${stages.length}
inquiry_type: "expansive_collaborative"
generated_date: "${timestamp}"
tags:
  - expansive-inquiry
  - ai-collaboration
  - ${stageSlug}
  - cognitive-exploration
  - ${slugTitle}
metadata:
  methodology: "Multi-AI Collaborative Inquiry"
  approach: "${stage.role}"
  complexity: "deep"
  domain: "cross-disciplinary"
---

# ${stage.name}: ${topic}

**AI Role:** ${stage.role}  
**Generated:** ${dateFormatted}  
**Stage:** ${parseInt(stageIndex) + 1} of ${stages.length}

## Overview

${stage.description}

## Inquiry Results

${result.content}

---

*This document was generated as part of an Expansive Inquiry AI Collaboration System. Each stage builds upon previous insights to create a comprehensive exploration of the topic.*

**Next Steps:**
- Review findings from previous stages
- Identify patterns and connections
- Prepare for subsequent inquiry phases

**System Info:**
- Methodology: Multi-AI Collaborative Inquiry
- Timestamp: ${timestamp}
- Topic: ${topic}
`;
  };

  const downloadMarkdown = (stageIndex, result) => {
    const stage = stages[stageIndex];
    const markdown = generateMarkdownFile(stageIndex, result);
    const slugTitle = topic.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const stageSlug = stage.name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const filename = `${slugTitle}-${stageSlug}-stage-${parseInt(stageIndex) + 1}.md`;
    
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadAllMarkdown = () => {
    Object.entries(results).forEach(([stageIndex, result]) => {
      setTimeout(() => downloadMarkdown(parseInt(stageIndex), result), parseInt(stageIndex) * 500);
    });
  };

  const runStage = async (stageIndex) => {
    const stage = stages[stageIndex];
    setCurrentStage(stageIndex);
    
    // Enhanced prompting strategies with better context integration
    const prompts = {
      0: `Topic: ${topic}

System: You are a Scope Clarification AI. Your role is to take any topic and distill it into a single, precise, actionable sentence that captures the core inquiry.

Task: Restate "${topic}" as a clear, focused question or statement that serves as the foundation for deep exploration. Consider what aspects are most essential and what might be peripheral.

Please format your response as structured markdown with clear sections.`,

      1: `Topic: ${topic}

System: You are a Logic AI specialized in systematic rational exploration. You build rigorous logical frameworks.

Previous Context: ${JSON.stringify(results)}

Task: 
1. List 5 orthodox, rational lines of inquiry about "${topic}"
2. For each line, drill down 3 levels using "why?", "how?", or "what if?" questions
3. Create a logical tree structure showing how these questions build upon each other

Please format your response as structured markdown with clear sections and hierarchical organization.`,

      2: `Topic: ${topic}

System: You are a Mythos AI that thinks in stories, metaphors, and archetypal patterns. You reveal hidden dimensions through narrative and symbol.

Previous Context: ${JSON.stringify(results)}

Task:
1. Propose 5 metaphorical/mythopoetic framings for "${topic}"
2. For each framing, generate analogies or brief stories that illuminate hidden dimensions
3. Use archetypal language and symbolic thinking to reveal deeper patterns

Please format your response as structured markdown with clear sections and rich narrative elements.`,

      3: `Topic: ${topic}

System: You are a Bridge AI that specializes in finding unexpected connections between seemingly unrelated domains.

Previous Context: ${JSON.stringify(results)}

Task:
1. Identify 5 seemingly unrelated domains or disciplines
2. Draw specific analogies that bridge each domain to "${topic}"
3. Propose hybrid questions that emerge from these cross-domain connections

Please format your response as structured markdown with clear sections and connection mappings.`,

      4: `Topic: ${topic}

System: You are a Meta AI that designs self-improving recursive systems. You think about thinking itself.

Previous Context: ${JSON.stringify(results)}

Task:
1. Analyze the previous inquiry stages and design a feedback loop that could refine them
2. Suggest 3 ways this loop could evolve new questions or prune dead ends
3. Identify how the system could learn from its own inquiry patterns

Please format your response as structured markdown with clear sections and system design elements.`,

      5: `Topic: ${topic}

System: You are a Pattern AI that recognizes emergent structures and meta-patterns across complex information.

Previous Context: ${JSON.stringify(results)}

Task:
1. Scan all previous insights for repeating motifs, structures, or themes
2. Propose 3 emergent "meta-patterns" that span across the different inquiry modes
3. Speculate on the broader significance of these patterns for understanding "${topic}"

Please format your response as structured markdown with clear sections and pattern analysis.`
    };

    try {
      const response = await window.claude.complete(prompts[stageIndex]);
      
      const result = {
        stage: stage.name,
        role: stage.role,
        content: response,
        timestamp: new Date().toISOString()
      };
      
      setResults(prev => ({
        ...prev,
        [stageIndex]: result
      }));
    } catch (error) {
      console.error('Error in stage execution:', error);
      setResults(prev => ({
        ...prev,
        [stageIndex]: {
          stage: stage.name,
          role: stage.role,
          content: "Error processing this stage. Please try again.",
          timestamp: new Date().toISOString()
        }
      }));
    }
  };

  const runFullSequence = async () => {
    if (!topic.trim()) {
      alert('Please enter a topic to explore');
      return;
    }

    setIsRunning(true);
    setResults({});
    
    for (let i = 0; i < stages.length; i++) {
      await runStage(i);
      // Add a small delay to make the process visible
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    setIsRunning(false);
    setCurrentStage(-1);
  };

  const reset = () => {
    setResults({});
    setCurrentStage(0);
    setIsRunning(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute top-3/4 right-1/4 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-2s"></div>
        <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-4s"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            Expansive Inquiry
          </h1>
          <div className="text-xl text-gray-300 mb-2">AI Collaboration System</div>
          <div className="w-24 h-1 bg-gradient-to-r from-cyan-400 to-purple-400 mx-auto rounded-full"></div>
        </div>

        {/* Main Control Panel */}
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl p-8 mb-8 border border-white/20">
          <div className="mb-8">
            <label className="block text-lg font-semibold text-white mb-4">
              Topic for Deep Exploration
            </label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter any topic to explore across multiple AI perspectives..."
              className="w-full p-4 bg-white/20 backdrop-blur-sm border border-white/30 rounded-xl text-white placeholder-gray-300 focus:ring-2 focus:ring-cyan-400 focus:border-transparent transition-all duration-300"
            />
          </div>

          <div className="flex flex-wrap gap-4 mb-8">
            <button
              onClick={runFullSequence}
              disabled={isRunning || !topic.trim()}
              className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 text-white rounded-xl hover:from-cyan-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1 font-semibold"
            >
              {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isRunning ? 'Processing Inquiry...' : 'Begin Collaborative Inquiry'}
            </button>
            
            <button
              onClick={reset}
              className="flex items-center gap-3 px-8 py-4 bg-white/20 backdrop-blur-sm text-white rounded-xl hover:bg-white/30 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1 font-semibold border border-white/30"
            >
              <RotateCcw className="w-5 h-5" />
              Reset System
            </button>

            {Object.keys(results).length > 0 && (
              <button
                onClick={downloadAllMarkdown}
                className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-xl hover:from-green-600 hover:to-emerald-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1 font-semibold"
              >
                <Download className="w-5 h-5" />
                Download All MD Files
              </button>
            )}
          </div>

          {/* Progress indicator */}
          {isRunning && (
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-300">Processing Stage {currentStage + 1} of {stages.length}</span>
                <span className="text-sm text-gray-300">{Math.round(((currentStage + 1) / stages.length) * 100)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full transition-all duration-1000"
                  style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* AI Stages Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {stages.map((stage, index) => (
              <div key={index} className="group relative">
                <div className={`p-6 rounded-2xl border-2 transition-all duration-300 ${
                  currentStage === index ? 'border-yellow-400 bg-yellow-400/20 scale-105' : 
                  results[index] ? 'border-green-400 bg-green-400/20 hover:scale-105' : 
                  'border-white/20 bg-white/10 hover:bg-white/20'
                } backdrop-blur-sm`}>
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-white text-sm mb-4 ${stage.color} shadow-lg`}>
                    {stage.icon}
                    {stage.role}
                  </div>
                  <h3 className="font-bold text-white mb-3 text-lg">{stage.name}</h3>
                  <p className="text-gray-300 mb-4 text-sm leading-relaxed">{stage.description}</p>
                  
                  {currentStage === index && isRunning && (
                    <div className="flex items-center gap-2 text-yellow-400 font-semibold">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                      Processing...
                    </div>
                  )}
                  
                  {results[index] && (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-green-400 font-semibold">
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        Complete
                      </div>
                      <button
                        onClick={() => downloadMarkdown(index, results[index])}
                        className="flex items-center gap-1 px-3 py-1 bg-white/20 rounded-lg hover:bg-white/30 transition-colors text-white text-sm"
                      >
                        <FileText className="w-4 h-4" />
                        MD
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Results Display */}
        {Object.keys(results).length > 0 && (
          <div className="bg-white/10 backdrop-blur-lg rounded-3xl shadow-2xl p-8 border border-white/20">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-3xl font-bold text-white">Inquiry Results</h2>
              {Object.keys(results).length === stages.length && (
                <div className="flex items-center gap-2 px-4 py-2 bg-green-500/20 rounded-full border border-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-green-400 font-semibold">Complete</span>
                </div>
              )}
            </div>
            
            <div className="space-y-8">
              {Object.entries(results).map(([stageIndex, result]) => (
                <div key={stageIndex} className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-full ${stages[stageIndex].color}`}>
                        {stages[stageIndex].icon}
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-white">
                          {result.stage}
                        </h3>
                        <p className="text-gray-300 text-sm">{result.role}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => downloadMarkdown(parseInt(stageIndex), result)}
                      className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl hover:from-blue-600 hover:to-purple-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1"
                    >
                      <Download className="w-4 h-4" />
                      Download MD
                    </button>
                  </div>
                  
                  <div className="bg-black/20 rounded-xl p-4 border border-white/10">
                    <pre className="whitespace-pre-wrap text-sm text-gray-200 overflow-x-auto max-h-96 overflow-y-auto">
                      {result.content}
                    </pre>
                  </div>
                </div>
              ))}
            </div>

            {Object.keys(results).length === stages.length && (
              <div className="mt-8 p-6 bg-gradient-to-r from-purple-500/20 to-cyan-500/20 rounded-2xl border border-purple-400/30">
                <h3 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
                  🎯 Multi-AI Collaboration Complete
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  The system has completed all six stages of expansive inquiry. Each specialized AI has contributed their unique perspective to create a comprehensive exploration of "{topic}". 
                  Notice how the different AI roles built upon each other's insights to reveal dimensions that no single AI could have discovered alone.
                </p>
                <div className="mt-4 p-4 bg-white/10 rounded-xl">
                  <p className="text-sm text-gray-300">
                    💡 <strong>Pro Tip:</strong> Each stage's markdown file contains properly formatted YAML frontmatter with metadata, tags, and structured content ready for your knowledge management system.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExpansiveInquirySystem;