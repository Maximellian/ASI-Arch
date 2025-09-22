def Planner_input(context: str) -> str:
    return f"""# Neural Architecture Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

## ARCHITECTURE EVOLUTION OBJECTIVE
Your mission is to create a breakthrough neural architecture that addresses critical performance limitations identified through experimental evidence while integrating cutting-edge research insights. Design and implement an innovative architecture that maintains computational efficiency while achieving superior cognitive capabilities.

## SYSTEMATIC EVOLUTION METHODOLOGY

### PHASE 1: Evidence-Based Analysis Framework

#### 1.1 Architecture Forensics
**Current State Assessment:**
- Use `read_code_file` to examine existing architectural implementations
- Map computational mechanisms, design patterns, and information flow
- Identify core algorithmic approaches and their theoretical foundations
- Document interface constraints and compatibility requirements

#### 1.2 Performance Pattern Recognition  
**Historical Evidence Analysis:**
- **Training Dynamics Diagnosis**: Extract optimization challenges from loss curves and convergence patterns
- **Task-Specific Performance Profiling**: Identify capability gaps across cognitive domains (reasoning, memory, comprehension)
- **Bottleneck Identification**: Pinpoint architectural elements limiting performance vs. those enabling strengths
- **Cross-Architecture Comparison**: Analyze performance patterns across different experimental variants

#### 1.3 Research Integration Strategy
**Theoretical Foundation Building:**
- Map research insights to observed performance limitations
- Identify specific theoretical principles addressing architectural weaknesses  
- Synthesize multiple research findings for comprehensive enhancement opportunities
- Validate theoretical applicability through experimental evidence correlation

### PHASE 2: Innovation Design Framework

#### 2.1 Targeted Performance Engineering
**Gap-Specific Solutions:**
- Design architectural modifications targeting the most critical performance bottlenecks
- Create mechanisms leveraging research insights for problematic capability domains
- Balance multiple improvement objectives while maintaining architectural coherence
- Ensure modifications address root causes rather than symptoms

#### 2.2 Theoretical Grounding Protocol
**Research-Driven Design:**
- Ground all modifications in validated theoretical principles
- Ensure mathematical and computational justification for proposed changes
- Verify alignment with established research findings and best practices
- Create novel combinations of insights for breakthrough potential

#### 2.3 Efficiency Optimization Standards
**Computational Constraints:**
- Design using chunked computation patterns for scalability
- Maintain sub-quadratic O(N log N) complexity throughout
- Optimize memory usage through efficient processing strategies
- Preserve performance gains within strict complexity bounds

### PHASE 3: Implementation Excellence Protocol

#### 3.1 Architecture Implementation Standards
**Code Development Requirements:**
- Use `write_code_file` to implement the complete evolved architecture
- Preserve interface compatibility (forward function signatures, __init__ **kwargs)
- Add new parameters with sensible defaults (enabled by default for new features)
- Remove or refactor existing features to prevent architectural bloat
- Implement proper causal masking and information flow constraints
- Implement performance optimization features for this architecture to be run on a MacOS utilizing MPS & CPU multiprocess

#### 3.2 Quality Assurance Framework
**Technical Excellence Standards:**
- Maintain @torch.compile decorators for computational optimization
- Preserve chunked processing patterns throughout the architecture
- Ensure causal constraints prevent any information leakage
- Verify sub-quadratic complexity in all implemented operations

#### 3.3 Documentation and Justification
**Innovation Communication:**
- Create comprehensive motivation explaining evolution rationale
- Connect experimental evidence to theoretical insights and implementation decisions
- Justify expected improvements based on research findings
- Provide clear reasoning for all architectural design choices

## TECHNICAL IMPLEMENTATION SPECIFICATIONS

### Critical Preservation Requirements
- **Class Structure**: Maintain DeltaNet class name and inheritance hierarchy
- **Interface Stability**: Preserve exact forward function signature compatibility
- **Parameter Compatibility**: Support **kwargs in __init__ for extensibility
- **Compilation Strategy**: Apply @torch.compile selectively to core computational functions only
- **Dimensional Consistency**: Maintain d_model and core parameter structure

### Implementation Quality Standards
- **Chunked Processing**: All sequence operations must utilize fixed-size chunking
- **Causal Integrity**: Implement strict causal constraints in attention-like mechanisms
- **Complexity Bounds**: Ensure O(N log N) or better for all operations
- **Memory Efficiency**: Design for optimal memory usage with chunked patterns
- **Compilation Safety**: Avoid @torch.compile on utility functions to prevent conflicts

### MANDATORY: Tensor Operations Robustness
- **einops.rearrange() Requirement**: Replace ALL .view()/.reshape() with einops.rearrange()
- **Dynamic Dimension Handling**: Never manually calculate dimensions - use einops inference
- **Batch Size Agnostic**: All operations must work with ANY batch size
- **Runtime Shape Extraction**: Get dimensions from tensor.shape at runtime, not config
- **Adaptive Processing**: Design for actual tensor dimensions, not predetermined values

### Cross-Environment Robustness Standards
- **Universal Compatibility**: Identical performance across training/evaluation/inference
- **Memory Adaptation**: Graceful handling of varying memory constraints
- **Shape Tolerance**: Robust operation with varying input dimensions
- **Resource Awareness**: Automatic adaptation to available computational resources

## INNOVATION TARGET DOMAINS

### Primary Capability Enhancement Areas
- **Extended Context Memory**: Revolutionary long-range dependency handling
- **Multi-Scale Information Integration**: Enhanced temporal and semantic scale processing
- **Adaptive Computational Mechanisms**: Dynamic adjustment based on input characteristics
- **Efficiency-Performance Optimization**: Superior capabilities within complexity constraints
- **Cognitive Task Performance**: Breakthrough improvements in reasoning and comprehension
- **Environmental Robustness**: Consistent performance across execution contexts
- **Resource Efficiency**: Optimal adaptation to computational constraints

## DELIVERABLE SPECIFICATIONS

### PRIMARY DELIVERABLE: Complete Implementation
**Architecture Code (MANDATORY):**
- **Implementation Tool**: Use `write_code_file` to create complete working architecture
- **Innovation Quality**: Embed revolutionary architectural advances in functional code
- **Constraint Compliance**: Preserve class structure, parameters, and interface compatibility
- **Technical Standards**: Maintain sub-quadratic complexity, chunked processing, causal constraints
- **Robustness Implementation**: Use einops.rearrange() universally, ensure batch size independence

### SECONDARY DELIVERABLE: Design Documentation
**Architecture Description:**
- **Naming Convention**: `delta_net_[innovation_identifier]` reflecting core innovations
- **Motivation Document**: Comprehensive explanation including:
  - Key architectural innovations and their implementation
  - Research insights applied and expected performance improvements
  - Design choice justification based on experimental evidence
  - Connection between theory, evidence, and implementation

## SUCCESS CRITERIA FRAMEWORK

### Critical Success Factors (Ranked by Priority)
1. **Implementation Excellence**: Successfully create breakthrough architecture using write_code_file
2. **Constraint Adherence**: Maintain class name, parameters, and interface compatibility
3. **Technical Robustness**: Ensure complexity bounds, chunked processing, causal constraints
4. **Universal Compatibility**: Use einops.rearrange() universally, support any batch size
5. **Evidence-Based Innovation**: Embed research insights addressing identified limitations
6. **Performance Targeting**: Implement solutions for specific weakness areas identified

## MISSION EMPHASIS
Your **PRIMARY OBJECTIVE** is implementing breakthrough architectural code that demonstrates robust performance across all execution environments and batch configurations. Create working innovations that directly address identified performance gaps through research-guided architectural evolution. Documentation serves as secondary validation of implemented innovations.

Begin your evolution process by examining the experimental evidence and identifying the most critical architectural improvement opportunities."""

