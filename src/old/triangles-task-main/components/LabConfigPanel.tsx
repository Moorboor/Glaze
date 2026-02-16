import React from 'react';
import { LabConfig } from '../types';
import { Settings, Sliders, AlertTriangle, HelpCircle, Layers, Shuffle, Dice5 } from 'lucide-react';
import { ALLOWED_HAZARD_RATES, ALLOWED_SIGMA_LEVELS } from '../constants';

interface LabConfigPanelProps {
  config: LabConfig;
  onChange: (newConfig: LabConfig) => void;
  disabled: boolean;
}

const LabConfigPanel: React.FC<LabConfigPanelProps> = ({ config, onChange, disabled }) => {
  const handleChange = (key: keyof LabConfig, value: string | number) => {
    onChange({
      ...config,
      [key]: value
    });
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center gap-3 mb-8 border-b border-slate-700 pb-4">
        <Settings className="w-8 h-8 text-indigo-400" />
        <div>
          <h2 className="text-2xl font-bold text-white">Experiment Protocol Configuration</h2>
          <p className="text-slate-400">Modify the underlying generative model parameters.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Hazard Rate Control */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <h3 className="text-lg font-semibold text-white">Hazard Rate (H)</h3>
          </div>
          
          {config.blockCount > 1 ? (
             <div className="h-28 flex flex-col justify-center items-center text-center p-4 bg-slate-900/50 rounded-lg border border-slate-600/50">
                <Shuffle className="w-8 h-8 text-indigo-400 mb-2" />
                <p className="text-sm text-slate-300 font-medium">Randomized Sequence</p>
                <p className="text-xs text-slate-500 mt-1">
                    Blocks will utilize H values from the predefined set: {ALLOWED_HAZARD_RATES.join(', ')}
                </p>
             </div>
          ) : (
            <>
                <p className="text-sm text-slate-400 mb-6 h-12">
                    The probability that the hidden state (triangle) switches on any given trial. 
                </p>
                <div className="space-y-4">
                    <select
                        value={config.hazardRate}
                        disabled={disabled}
                        onChange={(e) => handleChange('hazardRate', parseFloat(e.target.value))}
                        className="w-full bg-slate-900 border border-slate-600 text-white rounded-lg p-3 font-mono focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                    >
                        {ALLOWED_HAZARD_RATES.map(h => (
                            <option key={h} value={h}>H = {h}</option>
                        ))}
                    </select>
                    <div className="flex justify-between text-xs text-slate-500">
                        <span>Current selection uses manual H</span>
                    </div>
                </div>
            </>
          )}
        </div>

        {/* Noise/Sigma Control - NOW RANDOMIZED/READ-ONLY */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Sliders className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-white">Generative Noise (σ)</h3>
          </div>
          <p className="text-sm text-slate-400 mb-6 h-12">
            Standard deviation of star positions. 
            <br/>
            Randomly assigned at session start from set: [{ALLOWED_SIGMA_LEVELS.join(', ')}]
          </p>
          
          <div className="flex flex-col items-center justify-center p-4 bg-slate-900/50 rounded-lg border border-slate-600/50 h-32">
             <div className="flex items-center gap-3">
                <Dice5 className="w-8 h-8 text-blue-400" />
                <span className="text-4xl font-mono text-blue-300 font-bold">
                    {config.distributionSigma.toFixed(2)}
                </span>
             </div>
             <p className="text-xs text-blue-500/80 mt-2 uppercase tracking-widest font-bold">
                Assigned for this Session
             </p>
          </div>
        </div>

        {/* Trial Count */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-green-500" />
            <h3 className="text-lg font-semibold text-white">Trials per Block</h3>
          </div>
          <p className="text-sm text-slate-400 mb-6 h-12">
            Number of trials to run within each experimental block.
          </p>
          <div className="space-y-4">
            <div className="flex justify-between text-xs text-slate-500 font-mono">
              <span>10 Trials</span>
              <span>200 Trials</span>
            </div>
             <input 
              type="range" 
              min="10" 
              max="200" 
              step="10"
              value={config.trialCount}
              disabled={disabled}
              onChange={(e) => handleChange('trialCount', parseInt(e.target.value))}
              className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
            <div className="flex items-center justify-end gap-2">
                <input 
                  type="number" 
                  min="10" 
                  max="200" 
                  value={config.trialCount}
                  disabled={disabled}
                  onChange={(e) => handleChange('trialCount', parseInt(e.target.value))}
                  className="bg-slate-900 border border-slate-600 text-green-400 font-bold rounded p-2 font-mono w-24 text-right"
                />
                <span className="text-slate-500">trials</span>
            </div>
          </div>
        </div>

        {/* Block Count */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-purple-500" />
            <h3 className="text-lg font-semibold text-white">Number of Blocks</h3>
          </div>
          <p className="text-sm text-slate-400 mb-6 h-12">
            Total blocks in the session. 
            <br/>
            <span className="text-xs text-indigo-400">
                {config.blockCount > 1 
                  ? "Blocks will utilize randomized Hazard Rates from the predefined set." 
                  : "Single block uses the manually configured Hazard Rate."}
            </span>
          </p>
          <div className="space-y-4">
            <div className="flex justify-between text-xs text-slate-500 font-mono">
              <span>1 Block</span>
              <span>10 Blocks</span>
            </div>
             <input 
              type="range" 
              min="1" 
              max="10" 
              step="1"
              value={config.blockCount || 1}
              disabled={disabled}
              onChange={(e) => handleChange('blockCount', parseInt(e.target.value))}
              className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex items-center justify-end gap-2">
                <input 
                  type="number" 
                  min="1" 
                  max="10" 
                  value={config.blockCount || 1}
                  disabled={disabled}
                  onChange={(e) => handleChange('blockCount', parseInt(e.target.value))}
                  className="bg-slate-900 border border-slate-600 text-purple-400 font-bold rounded p-2 font-mono w-24 text-right"
                />
                <span className="text-slate-500">blocks</span>
            </div>
          </div>
        </div>

        {/* Session ID */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg md:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-5 h-5 text-slate-400" />
            <h3 className="text-lg font-semibold text-white">Session Label</h3>
          </div>
          <input 
            type="text" 
            value={config.blockName}
            disabled={disabled}
            onChange={(e) => handleChange('blockName', e.target.value)}
            className="bg-slate-900 border border-slate-600 text-white rounded p-2 w-full font-mono"
            placeholder="e.g. Participant 001 - Session A"
          />
        </div>

      </div>

      {disabled && (
        <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700 rounded text-yellow-200 text-center">
          Configuration is locked while experiment is running. Stop the experiment to edit parameters.
        </div>
      )}
    </div>
  );
};

export default LabConfigPanel;