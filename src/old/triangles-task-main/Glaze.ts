/**
 * Types representing a single trial in the Triangles task.
 */
export interface TrialData {
  /** The raw horizontal position of the star (e.g., -100 to 100) */
  starPosition: number;
  /** The user's choice: 0 for Left (negative), 1 for Right (positive) */
  userChoice: 0 | 1;
}

/**
 * Configuration for the task geometry to calculate LLR.
 * These values depend on your specific canvas/coordinate system.
 */
export interface TaskConfig {
  /** The x-coordinate of the right triangle center (e.g., 80) */
  meanPosition: number; 
  /** The standard deviation of the generative cloud (e.g., 20) */
  stdDev: number;       
}

export class GlazeModel {
  
  /**
   * GLAZE EQUATION 2: The Time-Varying Prior Expectation function (Psi).
   * This calculates the prior belief for the current step based on the 
   * posterior of the previous step and the hazard rate.
   * 
   * Source: Glaze et al. (2015), Eq 2 [2].
   * 
   * @param Ln_prev The belief (log-posterior odds) from the previous trial.
   * @param H The subjective Hazard Rate (probability of switching 0 to 1).
   */
  private static psi(Ln_prev: number, H: number): number {
    // Clamp H to avoid division by zero or log(0) errors
    const h = Math.max(0.000001, Math.min(0.999999, H));
    
    // The term (1-H)/H represents the prior odds of stability
    const stabilityRatio = (1 - h) / h;

    // Equation 2: 
    // ψ(L) = L + log((1-H)/H + exp(-L)) - log((1-H)/H + exp(L))
    const term1 = Math.log(stabilityRatio + Math.exp(-Ln_prev));
    const term2 = Math.log(stabilityRatio + Math.exp(Ln_prev));

    return Ln_prev + term1 - term2;
  }

  /**
   * GLAZE EQUATION 1: The Belief Update Rule.
   * Updates the belief based on the prior expectation (psi) and new evidence (LLR).
   * 
   * Source: Glaze et al. (2015), Eq 1 [1].
   */
  private static update(Ln_prev: number, LLRn: number, H: number): number {
    // Ln = ψ(Ln-1, H) + LLRn
    return this.psi(Ln_prev, H) + LLRn;
  }

  /**
   * Helper: Calculates Log-Likelihood Ratio (LLR) from screen position.
   * Assuming two symmetric Gaussian distributions centered at +/- mean.
   * LLR(x) = (2 * mean * x) / variance
   */
  public static calculateLLR(x: number, config: TaskConfig): number {
    const variance = config.stdDev * config.stdDev;
    return (2 * config.meanPosition * x) / variance;
  }

  /**
   * Generates the full belief trajectory (L values) for a sequence of trials
   * given a specific Hazard Rate H.
   */
  public static getBeliefTrajectory(
    data: TrialData[], 
    config: TaskConfig, 
    H: number
  ): number[] {
    const beliefs: number[] = [];
    let currentL = 0; // Start with prior belief of 0 (neutral)

    for (const trial of data) {
      const llr = this.calculateLLR(trial.starPosition, config);
      
      // Apply Eq 1 & 2
      currentL = this.update(currentL, llr, H);
      
      beliefs.push(currentL);
    }
    return beliefs;
  }

  /**
   * Calculates the Negative Log Likelihood (NLL) of the user's choices
   * given a specific Hazard Rate H. We want to MINIMIZE this value.
   */
  public static calculateFitScore(
    data: TrialData[], 
    config: TaskConfig, 
    H: number
  ): number {
    const beliefs = this.getBeliefTrajectory(data, config, H);
    let totalNLL = 0;

    // Inverse temperature (beta) for the sigmoid choice function.
    // Ideally, this is fitted too, but for simple H estimation, 
    // fixing it (e.g., to 1.0 or based on performance) is a common approximation.
    const beta = 1.0; 

    for (let i = 0; i < data.length; i++) {
      const L = beliefs[i];
      const userChoice = data[i].userChoice;

      // Probability that the model chooses Right (1) given belief L
      // P(choice=1) = 1 / (1 + exp(-beta * L))
      const probRight = 1 / (1 + Math.exp(-beta * L));
      
      // Probability that the model chooses Left (0)
      const probLeft = 1 - probRight;

      // Calculate likelihood of the ACTUAL user choice
      const likelihood = userChoice === 1 ? probRight : probLeft;

      // Add to negative log likelihood (prevent log(0) with small epsilon)
      totalNLL -= Math.log(Math.max(1e-10, likelihood));
    }

    return totalNLL;
  }

  /**
   * Fits the Subjective Hazard Rate (H) by performing a grid search
   * to find the H that minimizes the Negative Log Likelihood of the user's choices.
   */
  public static fitSubjectiveH(
    data: TrialData[], 
    config: TaskConfig
  ): { fittedH: number; beliefTrajectory: number[] } {
    let bestH = 0.5;
    let minNLL = Infinity;

    // Grid search parameters
    const step = 0.01;
    const start = 0.001; // H cannot be exactly 0 or 1 in this model formulation
    const end = 0.999;

    for (let h = start; h <= end; h += step) {
      const nll = this.calculateFitScore(data, config, h);
      if (nll < minNLL) {
        minNLL = nll;
        bestH = h;
      }
    }

    // Return the best H and the trajectory it produced
    return {
      fittedH: bestH,
      beliefTrajectory: this.getBeliefTrajectory(data, config, bestH)
    };
  }
}