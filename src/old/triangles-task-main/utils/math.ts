import { LEFT_SOURCE_MEAN, RIGHT_SOURCE_MEAN, ALLOWED_HAZARD_RATES } from '../constants';
import { LabConfig, SessionBlock } from '../types';

/**
 * Box-Muller transform to generate standard normal distribution samples
 */
function randn_bm(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Generates the source sequence and star positions for a single block.
 * @param sigmaRatio The ratio of Standard Deviation to Distance (sigma / D).
 */
export const generateBlockTrials = (trialCount: number, hazardRate: number, sigmaRatio: number) => {
  const trials = [];
  
  // Randomly pick initial source (0 = left, 1 = right)
  let currentSource = Math.random() < 0.5 ? 0 : 1;
  
  // Calculate distance in relative units (0.75 - 0.25 = 0.5)
  const distance = RIGHT_SOURCE_MEAN - LEFT_SOURCE_MEAN;
  
  // Convert the Ratio (0.24, 0.33, etc) into the actual Standard Deviation
  const actualSigma = sigmaRatio * distance;

  for (let i = 0; i < trialCount; i++) {
    // Check for switch based on Hazard Rate (Bernoulli process)
    if (Math.random() < hazardRate) {
      currentSource = currentSource === 0 ? 1 : 0;
    }

    // Determine Mean based on source
    // X axis: shifts based on source
    const muX = currentSource === 0 ? LEFT_SOURCE_MEAN : RIGHT_SOURCE_MEAN;
    
    // Y axis: always centered
    const muY = 0.5;
    
    // Sample position from 2D Gaussian (isotropic sigma)
    // Limit to 0.05 - 0.95 to keep it mostly on screen
    let x = muX + randn_bm() * actualSigma;
    x = Math.max(0.05, Math.min(0.95, x));

    // REJECTION SAMPLING: Ensure x is not too close to 0.5 to avoid 0 points.
    // Reward is round(|x - 0.5| * 100). To get >= 1, we need |x - 0.5| >= 0.005.
    // We use 0.01 as a safety margin.
    while (Math.abs(x - 0.5) < 0.01) {
        x = muX + randn_bm() * actualSigma;
        x = Math.max(0.05, Math.min(0.95, x));
    }

    let y = muY + randn_bm() * actualSigma;
    y = Math.max(0.05, Math.min(0.95, y));

    trials.push({
      trialIndex: i,
      source: currentSource === 0 ? 'left' : 'right' as 'left' | 'right',
      x,
      y
    });
  }

  return trials;
};

/**
 * Generates a full session comprising multiple blocks.
 * 
 * Logic:
 * - If blockCount == 1: Use config.hazardRate (Manual mode).
 * - If blockCount > 1: Generate a sequence from ALLOWED_HAZARD_RATES.
 *   The first set of blocks (up to list length) are chosen randomly WITHOUT replacement.
 *   Any subsequent blocks are chosen randomly WITH replacement.
 */
export const generateSessionData = (config: LabConfig): SessionBlock[] => {
  const blocks: SessionBlock[] = [];

  // Single Block Mode: Respect user config directly
  if (config.blockCount === 1) {
    blocks.push({
      blockId: 1,
      hazardRate: config.hazardRate,
      trials: generateBlockTrials(config.trialCount, config.hazardRate, config.distributionSigma)
    });
    return blocks;
  }

  // Multi-block Mode: Randomized sequence
  const pool = [...ALLOWED_HAZARD_RATES];
  // Shuffle the pool for "without replacement" phase
  const shuffledPool = pool.sort(() => Math.random() - 0.5);

  for (let b = 0; b < config.blockCount; b++) {
    let h: number;

    if (b < shuffledPool.length) {
      // Phase 1: Without replacement
      h = shuffledPool[b];
    } else {
      // Phase 2: With replacement (random pick from original pool)
      h = pool[Math.floor(Math.random() * pool.length)];
    }

    const trials = generateBlockTrials(config.trialCount, h, config.distributionSigma);

    blocks.push({
      blockId: b + 1,
      hazardRate: h,
      trials: trials
    });
  }

  return blocks;
};

/**
 * Calculate reward magnitude based on distance from center (neutral point).
 * Only the horizontal component (X) is accounted for.
 */
export const calculateRewardMagnitude = (positionX: number): number => {
  const dist = Math.abs(positionX - 0.5);
  // Scale factor: max dist is approx 0.5. Let's make max reward ~50 points.
  return Math.round(dist * 100);
};