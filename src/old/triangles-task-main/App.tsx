import React, { useState, useMemo } from 'react';
import { AppState, ExperimentPhase, LabConfig, TabView, TrialResult } from './types';
import { generateSessionData } from './utils/math';
import TrianglesGame from './components/TrianglesGame';
import ResultsView from './components/ResultsView';
import LabConfigPanel from './components/LabConfigPanel';
import { ALLOWED_SIGMA_LEVELS } from './constants';
import { 
  Play, 
  BarChart2, 
  Settings, 
  FlaskConical, 
  StopCircle 
} from 'lucide-react';

const getRandomSigma = () => {
  return ALLOWED_SIGMA_LEVELS[Math.floor(Math.random() * ALLOWED_SIGMA_LEVELS.length)];
};

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    activeTab: 'experiment',
    phase: ExperimentPhase.IDLE,
    score: 0,
    history: [],
    config: {
      hazardRate: 0.05,
      distributionSigma: getRandomSigma(), // Initialize with a random sigma
      rewardScale: 100,
      trialCount: 40,
      blockCount: 4, 
      blockName: 'Session 1'
    }
  });

  // Generate full session data (multiple blocks) whenever config changes or we restart
  const sessionData = useMemo(() => {
    return generateSessionData(state.config);
  }, [state.config.trialCount, state.config.hazardRate, state.config.distributionSigma, state.config.blockCount, state.phase]);

  const handleStart = () => {
    // Pick a new random sigma for this new session
    const newSigma = getRandomSigma();
    
    setState(prev => ({ 
        ...prev, 
        phase: ExperimentPhase.RUNNING, 
        activeTab: 'experiment', 
        history: [], // Clear history on new run
        score: 0,
        config: {
            ...prev.config,
            distributionSigma: newSigma
        }
    }));
  };

  const handleStop = () => {
    setState(prev => ({ ...prev, phase: ExperimentPhase.FINISHED, activeTab: 'analysis' }));
  };

  const handleTrialComplete = (result: TrialResult) => {
    setState(prev => ({
      ...prev,
      score: prev.score + result.reward,
      history: [...prev.history, result]
    }));
  };

  const handleConfigChange = (newConfig: LabConfig) => {
    setState(prev => ({ ...prev, config: newConfig }));
  };

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200 overflow-hidden font-sans">
      
      {/* Sidebar Navigation */}
      <div className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col">
        <div className="p-6 border-b border-slate-800">
            <div className="flex items-center gap-2 text-indigo-400 mb-1">
                <FlaskConical className="w-6 h-6" />
                <span className="font-bold text-lg tracking-wider">Glaze 2015</span>
            </div>
            <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold">Triangles Task</div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
            <NavButton 
                active={state.activeTab === 'experiment'} 
                onClick={() => setState(s => ({...s, activeTab: 'experiment'}))}
                icon={<Play className="w-5 h-5" />}
                label="Experiment"
            />
            <NavButton 
                active={state.activeTab === 'analysis'} 
                onClick={() => setState(s => ({...s, activeTab: 'analysis'}))}
                icon={<BarChart2 className="w-5 h-5" />}
                label="Analysis"
            />
            <NavButton 
                active={state.activeTab === 'config'} 
                onClick={() => setState(s => ({...s, activeTab: 'config'}))}
                icon={<Settings className="w-5 h-5" />}
                label="Configuration"
            />
        </nav>

        {/* Quick Actions Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/50">
            {state.phase === ExperimentPhase.RUNNING ? (
                <button 
                    onClick={handleStop}
                    className="w-full flex items-center justify-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/50 p-3 rounded-lg transition-all"
                >
                    <StopCircle className="w-5 h-5" />
                    <span className="font-bold">Stop Session</span>
                </button>
            ) : (
                <button 
                    onClick={handleStart}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white p-3 rounded-lg shadow-lg hover:shadow-indigo-500/25 transition-all"
                >
                    <Play className="w-5 h-5 fill-current" />
                    <span className="font-bold">Run Session</span>
                </button>
            )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Bar */}
        <header className="h-16 border-b border-slate-800 bg-slate-900 flex items-center justify-between px-8">
            <h1 className="text-xl font-medium text-white">
                {state.activeTab === 'experiment' && 'Live Experiment'}
                {state.activeTab === 'analysis' && 'Data Analysis'}
                {state.activeTab === 'config' && 'Lab Configuration'}
            </h1>
            <div className="flex items-center gap-6">
                {state.phase !== ExperimentPhase.RUNNING && (
                    <div className="flex flex-col items-end">
                        <span className="text-xs text-slate-500 uppercase">Current Score</span>
                        <span className="text-xl font-mono font-bold text-yellow-500">{state.score}</span>
                    </div>
                )}
                {state.phase === ExperimentPhase.RUNNING && (
                    <div className="flex flex-col items-end opacity-30">
                        <span className="text-xs text-slate-500 uppercase">Current Score</span>
                        <span className="text-xl font-mono font-bold text-slate-400">---</span>
                    </div>
                )}
                <div className="h-8 w-px bg-slate-800"></div>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${state.phase === ExperimentPhase.RUNNING ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`}></div>
                    <span className="text-sm text-slate-400 font-medium">
                        {state.phase === ExperimentPhase.RUNNING ? 'ACTIVE' : 'IDLE'}
                    </span>
                </div>
            </div>
        </header>

        {/* Content Body */}
        <main className="flex-1 overflow-auto bg-slate-900/50 p-6 relative">
            {state.activeTab === 'experiment' && (
                state.phase === ExperimentPhase.RUNNING ? (
                    <TrianglesGame 
                        key="active-session"
                        config={state.config}
                        sessionData={sessionData}
                        onSessionComplete={handleStop}
                        onTrialComplete={handleTrialComplete}
                        currentTotalScore={state.score}
                    />
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500">
                        <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4">
                            <FlaskConical className="w-8 h-8 text-slate-600" />
                        </div>
                        <h2 className="text-xl font-semibold text-white mb-2">Ready to Experiment</h2>
                        <p className="max-w-md text-center mb-8">
                            Configure your parameters in the <span className="text-indigo-400">Configuration</span> tab.
                            <br/>
                            This session will run <span className="text-white font-bold">{state.config.blockCount} blocks</span> of <span className="text-white font-bold">{state.config.trialCount} trials</span> each.
                        </p>
                        <button 
                            onClick={handleStart}
                            className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-full font-bold transition-all shadow-lg shadow-indigo-500/25"
                        >
                            Start Session
                        </button>
                    </div>
                )
            )}

            {state.activeTab === 'analysis' && (
                <div className="max-w-6xl mx-auto">
                    {state.history.length > 0 ? (
                        <ResultsView history={state.history} onRestart={handleStart} />
                    ) : (
                        <div className="text-center py-20 text-slate-500">
                            <BarChart2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                            <p>No data collected yet. Run an experiment session first.</p>
                        </div>
                    )}
                </div>
            )}

            {state.activeTab === 'config' && (
                <LabConfigPanel 
                    config={state.config} 
                    onChange={handleConfigChange}
                    disabled={state.phase === ExperimentPhase.RUNNING}
                />
            )}
        </main>
      </div>
    </div>
  );
};

// Helper UI Component
const NavButton = ({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) => (
    <button 
        onClick={onClick}
        className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
            active 
                ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/50' 
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
        }`}
    >
        {icon}
        <span className="font-medium text-sm">{label}</span>
    </button>
);

export default App;