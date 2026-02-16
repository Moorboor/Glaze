export enum ExperimentPhase {
  IDLE = 'IDLE',
  RUNNING = 'RUNNING',
  FINISHED = 'FINISHED'
}

export type TabView = 'experiment' | 'analysis' | 'config';

export interface BlockConfig {
  id: number;
  trialCount: number;
  hazardRate: number;
  description: string;
}

export interface LabConfig {
  hazardRate: number;
  distributionSigma: number;
  rewardScale: number;
  trialCount: number;
  blockCount: number;
  blockName: string;
}

export interface TrialResult {
  trialNumber: number;
  blockNumber: number;
  trueSource: 'left' | 'right';
  starX: number; // 0 to 1 scale (Horizontal)
  starY: number; // 0 to 1 scale (Vertical)
  userChoice: 'left' | 'right' | null;
  isCorrect: boolean;
  reward: number;
  hazardRate: number;
  distributionSigma: number;
  reactionTime: number;
  estimatedH: number; // The subjective H fitted to data up to this trial within the block
}

export interface SessionBlock {
  blockId: number;
  hazardRate: number;
  trials: {
    trialIndex: number;
    source: 'left' | 'right';
    x: number;
    y: number;
  }[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface AppState {
  activeTab: TabView;
  config: LabConfig;
  phase: ExperimentPhase;
  score: number;
  history: TrialResult[];
}