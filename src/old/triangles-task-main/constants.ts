import { BlockConfig } from './types';

// Visual constants
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 400;
export const TRIANGLE_Y = 200;
export const TRIANGLE_SIZE = 40;

// The paper uses two overlapping distributions.
// We map the screen from 0.0 to 1.0.
// Setting means to 0.25 and 0.75 creates a distance D = 0.5.
// This corresponds to "distance D to 0.5 * screen_width".
export const LEFT_SOURCE_MEAN = 0.25;
export const RIGHT_SOURCE_MEAN = 0.75;

// Default/Fallback Sigma (Note: Actual sigma is now randomized per session)
// This is now interpreted as the Ratio (Sigma / Distance)
// 0.24 creates a mean LLR of ~9.
export const DISTRIBUTION_SIGMA = 0.24; 

// Allowed Hazard Rates for the experiment
export const ALLOWED_HAZARD_RATES = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95];

// Allowed Generative Noise (Sigma) levels
// These represent the ratio (Sigma / Distance) as per Glaze et al. (2015)
// 0.24 => Mean LLR ~ 9
// 0.33 => Mean LLR ~ 4.5
// 0.41 => Mean LLR ~ 3.33
export const ALLOWED_SIGMA_LEVELS = [0.24, 0.33, 0.41];

// Experiment Structure
// We define 3 blocks with varying hazard rates (H) to test adaptability
export const BLOCKS: BlockConfig[] = [
  {
    id: 1,
    trialCount: 30, // Short for demo purposes, would be ~100-1000 in real paper
    hazardRate: 0.05, // Very stable (H low)
    description: "Block 1: Initial calibration"
  },
  {
    id: 2,
    trialCount: 30,
    hazardRate: 0.9, // Very unstable (H high - switching almost every time)
    description: "Block 2"
  },
  {
    id: 3,
    trialCount: 30,
    hazardRate: 0.1, // Moderately stable
    description: "Block 3"
  }
];