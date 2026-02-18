import React, { useEffect, useState, useRef, useCallback } from 'react';
import { 
  CANVAS_WIDTH, 
  CANVAS_HEIGHT, 
  LEFT_SOURCE_MEAN, 
  RIGHT_SOURCE_MEAN, 
  TRIANGLE_SIZE 
} from '../constants';
import { calculateRewardMagnitude } from '../utils/math';
import { TrialResult, LabConfig, SessionBlock } from '../types';
import { GlazeModel, TrialData, TaskConfig } from '../Glaze';
import { Play } from 'lucide-react';

interface TrianglesGameProps {
  config: LabConfig;
  sessionData: SessionBlock[];
  onSessionComplete: (results: TrialResult[]) => void;
  onTrialComplete: (result: TrialResult) => void;
  currentTotalScore: number;
}

const TrianglesGame: React.FC<TrianglesGameProps> = ({ 
  config,
  sessionData, 
  onSessionComplete,
  onTrialComplete,
  currentTotalScore
}) => {
  const [blockIndex, setBlockIndex] = useState(0);
  const [trialIndex, setTrialIndex] = useState(0);
  
  // Game Flow States:
  // waiting: Init state
  // stimulus: Star is shown, waiting for input
  // feedback: Reward is shown
  // block_interstitial: Between blocks
  const [gameState, setGameState] = useState<'waiting' | 'stimulus' | 'feedback' | 'block_interstitial'>('waiting');
  
  const [lastResult, setLastResult] = useState<{correct: boolean, reward: number} | null>(null);
  const [blockScore, setBlockScore] = useState(0);
  
  // Track data within the current block for real-time fitting
  const [currentBlockHistory, setCurrentBlockHistory] = useState<TrialData[]>([]);

  const startTimeRef = useRef<number>(0);
  
  // Current active block data
  const currentBlock = sessionData[blockIndex];
  const currentTrials = currentBlock?.trials || [];

  // Determine if feedback should be shown for the current trial
  // First 40% and Last 20% of the block trials.
  const totalTrials = currentTrials.length;
  const feedbackLimit1 = Math.round(totalTrials * 0.4);
  const feedbackLimit2 = Math.round(totalTrials * 0.2);
  const showFeedback = trialIndex < feedbackLimit1 || trialIndex >= (totalTrials - feedbackLimit2);

  // Auto-start first trial of a block
  useEffect(() => {
    if (trialIndex === 0 && gameState === 'waiting' && currentBlock) {
      const timer = setTimeout(() => {
        setGameState('stimulus');
        startTimeRef.current = performance.now();
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [trialIndex, gameState, currentBlock]);

  const handleChoice = useCallback((choice: 'left' | 'right') => {
    if (gameState !== 'stimulus') return;

    const currentTrial = currentTrials[trialIndex];
    if (!currentTrial) return;

    const isCorrect = choice === currentTrial.source;
    
    // Reward depends only on Horizontal component (X)
    const magnitude = calculateRewardMagnitude(currentTrial.x);
    const reward = isCorrect ? magnitude : -magnitude;

    // 1. Prepare data for this trial
    const thisTrialData: TrialData = {
        starPosition: currentTrial.x - 0.5, // Centered at 0
        userChoice: choice === 'right' ? 1 : 0
    };

    // 2. Add to local block history to simulate "estimation up to now"
    const updatedBlockHistory = [...currentBlockHistory, thisTrialData];
    setCurrentBlockHistory(updatedBlockHistory);

    // 3. Perform Instantaneous Fit of Subjective H
    const distance = RIGHT_SOURCE_MEAN - LEFT_SOURCE_MEAN;
    const tConfig: TaskConfig = {
        meanPosition: distance / 2,
        stdDev: config.distributionSigma * distance
    };
    
    // Fit the model using all data in this block SO FAR
    const { fittedH } = GlazeModel.fitSubjectiveH(updatedBlockHistory, tConfig);

    const result: TrialResult = {
      trialNumber: trialIndex + 1,
      blockNumber: currentBlock.blockId,
      trueSource: currentTrial.source,
      starX: currentTrial.x,
      starY: currentTrial.y,
      userChoice: choice,
      isCorrect,
      reward,
      hazardRate: currentBlock.hazardRate, // Record the ACTUAL H of this block
      distributionSigma: config.distributionSigma, // Record the actual Sigma Ratio of this session
      reactionTime: performance.now() - startTimeRef.current,
      estimatedH: fittedH
    };

    setBlockScore(prev => prev + reward);
    onTrialComplete(result);
    setLastResult({ correct: isCorrect, reward });
    setGameState('feedback');

    // Timer to move to next trial or end block
    setTimeout(() => {
      if (trialIndex < currentTrials.length - 1) {
        // Next Trial
        setTrialIndex(prev => prev + 1);
        setGameState('stimulus');
        setLastResult(null);
        startTimeRef.current = performance.now();
      } else {
        // End of Block
        setGameState('block_interstitial');
      }
    }, 800); 
  }, [gameState, currentTrials, trialIndex, currentBlock, onTrialComplete, config.distributionSigma, currentBlockHistory]);

  const handleNextBlock = (e?: React.MouseEvent) => {
    // Safety: Remove focus from button to prevent Enter key from re-triggering it
    // if the user mashes Enter or if the button retains focus.
    if (e?.currentTarget instanceof HTMLElement) {
        e.currentTarget.blur();
    }

    if (blockIndex < sessionData.length - 1) {
      setBlockIndex(prev => prev + 1);
      setTrialIndex(0);
      setBlockScore(0);
      setLastResult(null);
      setCurrentBlockHistory([]); // Reset local history for new block
      setGameState('waiting');
    } else {
      // Session Finished
      onSessionComplete([]);
    }
  };

  // Keyboard support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Explicitly ignore Enter key to prevent browser default actions (like clicking focused buttons)
      // which was causing the session to end early or skip blocks.
      if (e.key === 'Enter') {
          e.preventDefault();
          return;
      }

      if (gameState === 'stimulus') {
        if (e.key === 'ArrowLeft') handleChoice('left');
        if (e.key === 'ArrowRight') handleChoice('right');
      } 
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleChoice, gameState]);

  // Derived positions
  const leftX = LEFT_SOURCE_MEAN * CANVAS_WIDTH;
  const rightX = RIGHT_SOURCE_MEAN * CANVAS_WIDTH;
  const centerY = CANVAS_HEIGHT / 2;
  const distance = RIGHT_SOURCE_MEAN - LEFT_SOURCE_MEAN; // relative 0-1
  
  // Calculate star position based on X and Y
  const currentStarX = currentTrials[trialIndex]?.x * CANVAS_WIDTH;
  const currentStarY = currentTrials[trialIndex]?.y * CANVAS_HEIGHT;

  if (!currentBlock) {
      return <div className="text-white">Initializing Session...</div>;
  }

  // INTERSTITIAL SCREEN
  if (gameState === 'block_interstitial') {
    const isLastBlock = blockIndex === sessionData.length - 1;
    return (
        <div className="flex flex-col items-center justify-center w-full h-full p-4">
             <div className="bg-slate-800 p-12 rounded-2xl shadow-2xl border border-slate-700 text-center max-w-lg">
                <h2 className="text-3xl font-bold text-white mb-2">Block {currentBlock.blockId} Complete!</h2>
                <div className="text-5xl font-mono text-yellow-400 mb-8 py-4 border-b border-slate-700">
                    {blockScore > 0 ? '+' : ''}{blockScore} pts
                </div>
                
                {isLastBlock ? (
                     <div className="space-y-4">
                        <p className="text-slate-300 mb-6">Session finished. Proceed to analysis.</p>
                        <button 
                            onClick={handleNextBlock}
                            className="px-8 py-4 bg-green-600 hover:bg-green-500 text-white font-bold rounded-xl text-lg shadow-lg transition-all w-full"
                        >
                            Finish Experiment
                        </button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <p className="text-slate-300 mb-6">
                            Take a short break. The next block will have a <span className="text-indigo-400 font-bold">different volatility</span>.
                            <br/>Adapt your strategy accordingly.
                        </p>
                        <button 
                            onClick={handleNextBlock}
                            className="flex items-center justify-center gap-2 px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl text-lg shadow-lg transition-all w-full"
                        >
                            <Play className="w-5 h-5 fill-current" />
                            Start Block {currentBlock.blockId + 1}
                        </button>
                    </div>
                )}
             </div>
        </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-4">
      {/* HUD */}
      <div className="w-full max-w-4xl flex justify-between items-center mb-4 text-slate-300">
        <div className="flex gap-4">
            <div className="text-sm font-mono bg-slate-800 px-3 py-1 rounded border border-slate-700 text-indigo-300">
            Block: {blockIndex + 1} / {sessionData.length}
            </div>
            <div className="text-sm font-mono bg-slate-800 px-3 py-1 rounded border border-slate-700">
            Trial: <span className="text-white font-bold">{trialIndex + 1}</span> / {currentTrials.length}
            </div>
        </div>
        
        <div className="text-xl font-bold bg-slate-800 px-4 py-1 rounded border border-slate-700 text-yellow-500 shadow-lg min-w-[140px] text-center">
          Score: {showFeedback ? currentTotalScore : '---'}
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="relative bg-slate-800 rounded-xl shadow-2xl border border-slate-700 overflow-hidden"
           style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}>
        
        <svg width="100%" height="100%" className="absolute top-0 left-0">
          <defs>
            <radialGradient id="greenCloudLeft" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
              <stop offset="0%" stopColor="rgba(34, 197, 94, 0.4)" />
              <stop offset="100%" stopColor="rgba(34, 197, 94, 0)" />
            </radialGradient>
            <radialGradient id="greenCloudRight" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
              <stop offset="0%" stopColor="rgba(34, 197, 94, 0.4)" />
              <stop offset="100%" stopColor="rgba(34, 197, 94, 0)" />
            </radialGradient>
          </defs>

          {/* Probability Clouds */}
          {/* Visual radius calculation:
              Standard Deviation = Ratio * Distance.
              rx = stdDev * width * scale_factor.
              Here scale_factor of 2.5 covers ~2.5 sigmas.
           */}
          <ellipse 
            cx={leftX} cy={centerY} 
            rx={CANVAS_WIDTH * (config.distributionSigma * distance * 2.5)} ry={CANVAS_HEIGHT * 0.4} 
            fill="url(#greenCloudLeft)" 
          />
          <ellipse 
            cx={rightX} cy={centerY} 
            rx={CANVAS_WIDTH * (config.distributionSigma * distance * 2.5)} ry={CANVAS_HEIGHT * 0.4} 
            fill="url(#greenCloudRight)" 
          />

          {/* Triangles */}
          <path 
            d={`M${leftX},${centerY - TRIANGLE_SIZE} L${leftX - TRIANGLE_SIZE},${centerY + TRIANGLE_SIZE} L${leftX + TRIANGLE_SIZE},${centerY + TRIANGLE_SIZE} Z`} 
            fill="none" stroke="#4ade80" strokeWidth="3"
            className="transition-all duration-300"
            style={{ 
              filter: showFeedback && lastResult && lastResult.correct && currentTrials[trialIndex].source === 'left' 
                ? 'drop-shadow(0 0 15px #4ade80)' 
                : 'none'
            }}
          />
          <path 
            d={`M${rightX},${centerY - TRIANGLE_SIZE} L${rightX - TRIANGLE_SIZE},${centerY + TRIANGLE_SIZE} L${rightX + TRIANGLE_SIZE},${centerY + TRIANGLE_SIZE} Z`} 
            fill="none" stroke="#4ade80" strokeWidth="3"
            className="transition-all duration-300"
            style={{ 
              filter: showFeedback && lastResult && lastResult.correct && currentTrials[trialIndex].source === 'right' 
                ? 'drop-shadow(0 0 15px #4ade80)' 
                : 'none'
            }}
          />

          {/* The Star Stimulus */}
          {gameState === 'stimulus' && (
            // Use translation based on both X and Y
            <g transform={`translate(${currentStarX}, ${currentStarY})`}>
              <polygon 
                points="0,-10 3,-3 10,-3 5,2 7,9 0,5 -7,9 -5,2 -10,-3 -3,-3"
                fill="#ef4444"
                className="animate-pulse"
              />
            </g>
          )}
        </svg>

        {/* Feedback Overlay */}
        {gameState === 'feedback' && showFeedback && lastResult && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            <div className={`text-5xl font-black tracking-tighter ${lastResult.correct ? 'text-green-400 drop-shadow-lg' : 'text-red-500 drop-shadow-lg'} animate-bounce`}>
              {lastResult.correct ? `+${lastResult.reward}` : `${lastResult.reward}`}
            </div>
          </div>
        )}

        {/* Interaction Areas */}
        <div className="absolute inset-0 flex">
          <button 
            className="w-1/2 h-full cursor-pointer hover:bg-white/5 active:bg-white/10 transition-colors focus:outline-none"
            onClick={() => handleChoice('left')}
            disabled={gameState !== 'stimulus'}
          />
          <button 
            className="w-1/2 h-full cursor-pointer hover:bg-white/5 active:bg-white/10 transition-colors focus:outline-none"
            onClick={() => handleChoice('right')}
            disabled={gameState !== 'stimulus'}
          />
        </div>
      </div>
    </div>
  );
};

export default TrianglesGame;