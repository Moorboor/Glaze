import React, { useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Scatter, ComposedChart, Area, ScatterChart
} from 'recharts';
import { TrialResult } from '../types';
import { GlazeModel, TrialData, TaskConfig } from '../Glaze';
import { RIGHT_SOURCE_MEAN, LEFT_SOURCE_MEAN, DISTRIBUTION_SIGMA } from '../constants';
import { Brain, Activity, TrendingUp, GitMerge, Database, FileText } from 'lucide-react';

interface ResultsViewProps {
  history: TrialResult[];
  onRestart: () => void;
}

// Helper for linear regression (y = mx + b)
const calculateRegression = (points: { x: number, y: number }[]) => {
    if (points.length < 2) return null;
    
    const n = points.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    points.forEach(p => {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumXX += p.x * p.x;
    });
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept };
};

const ResultsView: React.FC<ResultsViewProps> = ({ history, onRestart }) => {
  
  // 1. Process Data per Block
  const { blockAnalysis, sessionChartData, regressionLine } = useMemo(() => {
    if (history.length === 0) return { blockAnalysis: [], sessionChartData: [], regressionLine: [] };

    // Group trials by block
    const blocks: Record<number, TrialResult[]> = {};
    history.forEach(trial => {
        if (!blocks[trial.blockNumber]) blocks[trial.blockNumber] = [];
        blocks[trial.blockNumber].push(trial);
    });

    const analysisResults = [];
    let cumulativeTrialCount = 0;
    const fullTrajectory: { trial: number, belief: number }[] = [];

    // Analyze each block individually
    for (const blockIdStr of Object.keys(blocks)) {
        const blockTrials = blocks[parseInt(blockIdStr)];
        if (blockTrials.length === 0) continue;

        const firstTrial = blockTrials[0];
        
        // Task Config for this block
        const distance = RIGHT_SOURCE_MEAN - LEFT_SOURCE_MEAN;
        const actualSigmaRatio = firstTrial.distributionSigma || DISTRIBUTION_SIGMA;
        const tConfig: TaskConfig = {
            meanPosition: distance / 2,
            stdDev: actualSigmaRatio * distance
        };

        const trialData: TrialData[] = blockTrials.map(t => ({
            starPosition: t.starX - 0.5,
            userChoice: t.userChoice === 'right' ? 1 : 0
        }));

        // Fit Model for THIS block
        const fit = GlazeModel.fitSubjectiveH(trialData, tConfig);

        // Calculate specific stats
        const blockAccuracy = (blockTrials.filter(t => t.isCorrect).length / blockTrials.length * 100);
        
        analysisResults.push({
            blockId: parseInt(blockIdStr),
            trueH: firstTrial.hazardRate,
            fittedH: fit.fittedH,
            accuracy: blockAccuracy,
            trialCount: blockTrials.length
        });

        // Accumulate trajectory for the main chart
        fit.beliefTrajectory.forEach((belief, idx) => {
            fullTrajectory.push({
                trial: cumulativeTrialCount + idx + 1,
                belief: belief
            });
        });

        cumulativeTrialCount += blockTrials.length;
    }

    // Prepare Regression Data
    const regressionPoints = analysisResults.map(r => ({ x: r.trueH, y: r.fittedH }));
    const reg = calculateRegression(regressionPoints);
    let regLineData: { x: number, y: number }[] = [];
    
    if (reg) {
        // Create line points from min to max H
        const minH = Math.min(...regressionPoints.map(p => p.x), 0);
        const maxH = Math.max(...regressionPoints.map(p => p.x), 1);
        regLineData = [
            { x: minH, y: reg.slope * minH + reg.intercept },
            { x: maxH, y: reg.slope * maxH + reg.intercept }
        ];
    }

    // Merge Trajectory with original history for the big chart
    const combinedChartData = history.map((trial, idx) => ({
        trial: idx + 1,
        trueSourceVal: trial.trueSource === 'left' ? 0 : 1,
        userChoiceVal: trial.userChoice === 'left' ? 0 : 1,
        cumulativeReward: history.slice(0, idx + 1).reduce((acc, curr) => acc + curr.reward, 0),
        modelBelief: fullTrajectory[idx]?.belief || 0,
        // For coloring changes in block
        blockId: trial.blockNumber
    }));

    return { 
        blockAnalysis: analysisResults, 
        sessionChartData: combinedChartData,
        regressionLine: regLineData
    };
  }, [history]);

  // 3. Prepare Data for Psychometric Curve (Choice vs Evidence)
  const psychometricData = useMemo(() => {
    if (history.length === 0) return [];
    const binSize = 0.1;
    const bins: Record<string, { sumX: number, count: number, rightCount: number }> = {};

    history.forEach(t => {
      const x = t.starX - 0.5;
      const binKey = (Math.round(x / binSize) * binSize).toFixed(1);
      if (!bins[binKey]) bins[binKey] = { sumX: 0, count: 0, rightCount: 0 };
      bins[binKey].sumX += x;
      bins[binKey].count += 1;
      if (t.userChoice === 'right') bins[binKey].rightCount += 1;
    });

    return Object.values(bins)
      .map(b => ({
        evidence: b.sumX / b.count,
        probRight: b.rightCount / b.count
      }))
      .sort((a, b) => a.evidence - b.evidence);
  }, [history]);

  const totalScore = history.reduce((acc, curr) => acc + curr.reward, 0);
  const accuracy = (history.filter(h => h.isCorrect).length / history.length * 100).toFixed(1);
  
  // -- COPY DATA FUNCTIONS --

  const handleCopyChoiceData = () => {
    const headers = ['block_id', 'trial_index', 'hazard_rate', 'noise_sigma', 'LLR', 'choice', 'correct_side', 'reaction_time_ms', 'belief_L', 'subjective_h_snapshot'];
    
    // Constants for LLR calculation
    // Distance between centers = 0.75 - 0.25 = 0.5
    const distance = RIGHT_SOURCE_MEAN - LEFT_SOURCE_MEAN;
    const meanPos = distance / 2;

    const rows = history.map((t, index) => {
        // Calculate LLR based on Glaze et al. formula
        // LLR(x) = (2 * mean * x) / variance
        // x is centered position (0 is neutral)
        const x = t.starX - 0.5;
        const stdDev = t.distributionSigma * distance;
        const variance = stdDev * stdDev;
        const llr = (2 * meanPos * x) / variance;

        // Get belief from the calculated session data
        const beliefL = sessionChartData[index]?.modelBelief || 0;

        return [
            t.blockNumber,
            t.trialNumber,
            t.hazardRate,
            t.distributionSigma.toFixed(4),
            llr.toFixed(6),
            t.userChoice === 'right' ? 1 : 0,
            t.trueSource === 'right' ? 1 : 0,
            t.reactionTime.toFixed(0),
            beliefL.toFixed(6),
            t.estimatedH ? t.estimatedH.toFixed(4) : ''
        ];
    });
    
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    navigator.clipboard.writeText(csvContent).then(() => alert("Choice data copied to clipboard!"));
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-slate-800 rounded-xl shadow-2xl mb-12">
      
      {/* Header Stats */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 border-b border-slate-700 pb-6">
        <div>
           <h2 className="text-3xl font-bold text-white mb-2">Session Analysis</h2>
           <p className="text-slate-400">Performance metrics and Computational Model Fit</p>
        </div>
        <div className="flex flex-col md:flex-row gap-3 mt-4 md:mt-0">
             <button 
                onClick={handleCopyChoiceData}
                className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-medium transition-all text-sm border border-slate-600"
             >
                <Database className="w-4 h-4" />
                Copy Choice CSV
             </button>
             <button 
              onClick={onRestart}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold transition-all ml-2"
            >
              Start New Block
            </button>
        </div>
      </div>

      {/* Top Level Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-700/50 p-6 rounded-xl border border-slate-600 flex items-center justify-between">
            <div>
                <div className="flex items-center gap-3 mb-2">
                    <TrendingUp className="w-5 h-5 text-yellow-400" />
                    <span className="text-sm font-semibold text-slate-300">Final Score</span>
                </div>
                <div className="text-4xl font-mono text-white">{totalScore}</div>
            </div>
            <div className="h-12 w-12 rounded-full bg-yellow-500/10 flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-yellow-500" />
            </div>
        </div>
        
        <div className="bg-slate-700/50 p-6 rounded-xl border border-slate-600 flex items-center justify-between">
            <div>
                <div className="flex items-center gap-3 mb-2">
                    <Activity className="w-5 h-5 text-green-400" />
                    <span className="text-sm font-semibold text-slate-300">Overall Accuracy</span>
                </div>
                <div className="text-4xl font-mono text-white">{accuracy}%</div>
            </div>
            <div className="h-12 w-12 rounded-full bg-green-500/10 flex items-center justify-center">
                <Activity className="w-6 h-6 text-green-500" />
            </div>
        </div>
      </div>

      {/* --- NEW SECTION: Volatility Adaptation & Regression --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        
        {/* Left Col: Scatter Plot */}
        <div className="lg:col-span-1 bg-slate-900 rounded-xl p-6 border border-slate-700 flex flex-col">
            <h3 className="text-lg font-semibold text-slate-200 mb-2 flex items-center gap-2">
                <Brain className="w-5 h-5 text-pink-400" />
                Volatility Adaptation
            </h3>
            <p className="text-xs text-slate-500 mb-4">
                Compares the True Hazard Rate (Environment) vs. your Subjective Hazard Rate (Fitted).
            </p>
            {/* Aspect Square Container for the Chart */}
            <div className="w-full aspect-square relative max-h-[400px] mx-auto">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis 
                            type="number" dataKey="x" name="True H" unit="" domain={[0, 1]} 
                            stroke="#64748b" label={{ value: 'True H', position: 'insideBottom', offset: -5, fill: '#64748b' }}
                        />
                        <YAxis 
                            type="number" dataKey="y" name="Subjective H" unit="" domain={[0, 1]} 
                            stroke="#64748b" label={{ value: 'Subjective H', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                        />
                        <Tooltip 
                          cursor={{ strokeDasharray: '3 3' }} 
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                          itemStyle={{ color: '#e2e8f0' }}
                          labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
                          formatter={(value: number, name: string) => [
                            value.toFixed(2), 
                            name === 'x' ? 'True H' : (name === 'y' ? 'Subjective H' : name)
                          ]}
                        />
                        
                        {/* Identity Line (Perfect Calibration) */}
                        <Line 
                            dataKey="y" data={[{x:0, y:0}, {x:1, y:1}]} 
                            stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={1} dot={false} activeDot={false} isAnimationActive={false}
                            name="Perfect Calibration"
                        />
                        
                        {/* Regression Line */}
                        {regressionLine.length > 0 && (
                             <Line 
                                dataKey="y" data={regressionLine} 
                                stroke="#f472b6" strokeWidth={2} dot={false} activeDot={false} 
                                name="Trend"
                            />
                        )}

                        {/* Data Points */}
                        <Scatter name="Blocks" data={blockAnalysis.map(b => ({ x: b.trueH, y: b.fittedH, block: b.blockId }))} fill="#818cf8">
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </div>

        {/* Right Col: Block Breakdown Grid */}
        <div className="lg:col-span-2">
            <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                Block-by-Block Analysis
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 max-h-[400px] overflow-y-auto pr-2 scrollbar-hide">
                {blockAnalysis.map((block) => {
                    const diff = block.fittedH - block.trueH;
                    const diffColor = Math.abs(diff) < 0.1 ? 'text-green-400' : (diff > 0 ? 'text-orange-400' : 'text-blue-400');
                    return (
                        <div key={block.blockId} className="bg-slate-700/40 p-4 rounded-lg border border-slate-700 hover:bg-slate-700/60 transition-colors">
                            <div className="flex justify-between items-center mb-2 border-b border-slate-600/50 pb-2">
                                <span className="font-bold text-slate-300">Block {block.blockId}</span>
                                <span className="text-xs bg-slate-800 px-2 py-0.5 rounded text-slate-400">Acc: {block.accuracy.toFixed(0)}%</span>
                            </div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-500">True H:</span>
                                    <span className="font-mono text-slate-200">{block.trueH.toFixed(2)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-500">Fitted H:</span>
                                    <span className={`font-mono font-bold ${diffColor}`}>{block.fittedH.toFixed(2)}</span>
                                </div>
                                <div className="text-[10px] text-right pt-1 opacity-70">
                                    {Math.abs(diff) < 0.1 ? "Well Calibrated" : (diff > 0 ? "Overestimated" : "Underestimated")}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
      </div>

      {/* CHART 1: Belief Trajectory (The Glaze Model) */}
      <div className="mb-8 bg-slate-900 rounded-xl p-6 border border-slate-700">
        <h3 className="text-xl font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-pink-500" />
            Belief Dynamics (Concatenated Block Fits)
        </h3>
        <p className="text-sm text-slate-500 mb-6">
            Log-Likelihood Ratio accumulated over time. The model fit is calculated independently for each block to show how your strategy adapted.
        </p>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={sessionChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis dataKey="trial" stroke="#64748b" />
              <YAxis stroke="#64748b" label={{ value: 'Belief (Log Odds)', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              
              <Area 
                type="monotone" 
                dataKey="modelBelief" 
                stroke="#ec4899" 
                fill="url(#colorBelief)" 
                strokeWidth={2}
                fillOpacity={0.2}
                name="Model Belief (L)"
              />
              <defs>
                <linearGradient id="colorBelief" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                </linearGradient>
              </defs>

              <Scatter 
                dataKey={(d) => d.userChoiceVal === 1 ? 2 : -2} 
                fill="#fff" 
                name="User Choice"
                shape="circle"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* CHART 2: Psychometric Curve */}
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-700">
           <h3 className="text-xl font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <GitMerge className="w-5 h-5 text-blue-400 rotate-90" />
            Psychometric Curve (Aggregate)
           </h3>
           <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={psychometricData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                        dataKey="evidence" 
                        type="number" 
                        domain={[-0.5, 0.5]} 
                        stroke="#64748b"
                        label={{ value: 'Evidence (Position)', position: 'insideBottom', fill: '#64748b', dy: 10 }}
                    />
                    <YAxis 
                        domain={[0, 1]} 
                        stroke="#64748b"
                        label={{ value: 'P(Choose Right)', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                    />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value: number) => value.toFixed(2)}
                        labelFormatter={(label) => `Pos: ${Number(label).toFixed(2)}`}
                    />
                    <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="3 3" />
                    <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="3 3" />
                    
                    <Line 
                        type="monotone" 
                        dataKey="probRight" 
                        stroke="#3b82f6" 
                        strokeWidth={3} 
                        dot={false}
                        name="Fit"
                    />
                    <Scatter 
                        dataKey="probRight" 
                        fill="#60a5fa" 
                        r={4}
                        name="Actual Choices"
                    />
                </ComposedChart>
            </ResponsiveContainer>
           </div>
        </div>

        {/* CHART 3: Cumulative Score */}
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold text-slate-200 mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-yellow-500" />
                Performance History
            </h3>
            <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sessionChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="trial" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none' }} />
                <Line 
                    type="monotone" 
                    dataKey="cumulativeReward" 
                    stroke="#fbbf24" 
                    strokeWidth={2} 
                    dot={false} 
                    name="Score"
                />
                </LineChart>
            </ResponsiveContainer>
            </div>
        </div>
      </div>

    </div>
  );
};

export default ResultsView;