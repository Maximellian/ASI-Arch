def Analyzer_input(name: str, result: str, motivation: str, ref_context: str) -> str:
    """
    Creates a comprehensive prompt for experiment results analysis with emphasis
    on balanced assessment and critical analysis of architectural modifications.
    """
    return f"""# Analysis Request: Model {name}
## Resources:
- Results: `{result}`
- Code implementation: Use read_code_file tool to examine the architecture
- Design motivation: {motivation}
## Related Experiments for Ablation Study:
{ref_context}
**IMPORTANT:** The above related experiments represent either parent nodes (previous iterations that led to this design) or sibling nodes (alternative approaches explored from the same parent). Use these for ablation study analysis to understand:
- What specific changes differentiate the current experiment from its relatives
- Which architectural components are responsible for performance differences
- Whether the modifications represent genuine improvements or trade-offs
## Analysis Requirements:
Please read the results, examine the code implementation using read_code_file tool, and analyze the design motivation. Your analysis must include:
1. **MOTIVATION AND DESIGN EVALUATION**
   - Assess the theoretical soundness of the proposed architectural changes
   - Evaluate whether the code implementation correctly reflects the design intention
   - Identify any gaps between motivation and actual implementation
   - Judge the plausibility of expected improvements based on the architectural changes
2. **EXPERIMENTAL RESULTS ANALYSIS WITH ABLATION STUDY**
   - Summarize performance outcomes using task-descriptive language (e.g., "memory retention capability improved" rather than "Compress score increased to X")
   - Compare results with baseline models using clear improvement/degradation statements
   - **ABLATION ANALYSIS**: Compare with related experiments to identify:
     * Which specific architectural changes caused performance differences
     * Whether improvements are due to the intended modifications or other factors
     * Trade-offs introduced by each architectural component
   - Identify which cognitive capabilities were enhanced vs compromised
   - Provide an overall assessment of whether the modifications achieved their intended goals
3. **EXPECTATION VS REALITY COMPARISON**
   - Analyze whether experimental results align with the stated motivation and expected outcomes
   - Identify surprising results (both positive and negative) that weren't anticipated
   - Assess the accuracy of the design hypothesis based on empirical evidence
   - Determine if the architectural changes produced the predicted effects
   - **CROSS-EXPERIMENT VALIDATION**: Check if similar modifications in related experiments produced consistent effects
4. **THEORETICAL EXPLANATION WITH EVIDENCE**
   - Provide mechanistic explanations for observed performance patterns, supported by:
     * Specific code elements that caused the effects
     * Mathematical reasoning linking architectural changes to performance outcomes
     * Information-theoretic or computational arguments where applicable
   - **COMPARATIVE ANALYSIS**: Explain why this approach outperformed or underperformed relative experiments
   - For performance degradations: explain the precise mechanisms that undermined specific capabilities
   - For improvements: identify the architectural features responsible for enhanced performance
   - Connect theoretical predictions with empirical observations
5. **SYNTHESIS AND INSIGHTS**
   - Summarize key lessons learned about this type of architectural modification
   - **ABLATION INSIGHTS**: Based on comparison with related experiments, identify:
     * Essential vs. redundant architectural components
     * Optimal combinations of modifications
     * Architectural decisions that should be preserved or discarded in future iterations
   - Identify fundamental trade-offs revealed by the experiments
   - Provide actionable insights for future architectural design decisions
   - Suggest specific directions for addressing identified limitations
**Critical Analysis Standards:**
- Support all claims with specific evidence from code, results, or theoretical reasoning
- Use ablation study methodology: isolate the impact of individual changes by comparing with related experiments
- Be honest about failures and unexpected outcomes
- Focus on understanding WHY results occurred, not just WHAT happened
- Use capability-focused language rather than raw performance metrics
- Maintain scientific rigor in explanations and avoid speculation without evidence
- When analyzing improvements/degradations, always reference related experiments to validate conclusions
Your analysis should be thorough, evidence-based, and provide actionable insights for architectural innovation through systematic ablation study.
"""

# Recommendation (Do not alter the prompt unless optimizing for experiment automation):
# - Optionally, remind the analyst to reference experiment run IDs or output CSVs to link analysis with actual experiment logs/training runs.
# - Encourage noting any pipeline issues that may have influenced result integrity or experiment reproducibility.
