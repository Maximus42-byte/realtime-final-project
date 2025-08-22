# Gang Task Scheduling Analysis - Line-by-Line Code Explanation

## Overview

This document provides a detailed explanation of the gang task scheduling analysis implementation. The code implements various scheduling algorithms, allocation strategies, and response-time analysis methods for real-time multiprocessor systems.

## Import Statements (Lines 1-9)

```python
import numpy as np                      # For numerical computations
import matplotlib.pyplot as plt         # For plotting results
from dataclasses import dataclass       # For creating structured data classes
from typing import List, Tuple, Dict    # For type annotations
import random                          # For random number generation
from scipy.optimize import linprog      # For MILP solving (simplified as LP)
import time                            # For measuring computation time
from collections import defaultdict    # For efficient data aggregation
import seaborn as sns                  # For heatmap visualization
```

## Data Structures

### Bundle Class (Lines 11-15)

```python
@dataclass
class Bundle:
    """Represents a bundle (segment) within a task"""
    execution_time: float    # WCET of this bundle
    required_cores: int      # Number of cores needed for parallel execution
```

The `Bundle` class represents a segment of a task that requires a specific number of cores and has a worst-case execution time (WCET).

### Task Class (Lines 17-29)

```python
@dataclass
class Task:
    """Represents a real-time task with multiple bundles"""
    id: int                    # Unique task identifier
    period: float             # Task period (recurring interval)
    deadline: float           # Task deadline
    bundles: List[Bundle]     # List of task segments
    priority: float = 0.0     # Task priority (assigned by scheduling policy)
    
    def __post_init__(self):
        # Calculate total execution time by summing all bundle times
        self.wcet = sum(bundle.execution_time for bundle in self.bundles)
        # Find maximum parallelism requirement across all bundles
        self.max_parallelism = max(bundle.required_cores for bundle in self.bundles)
```

The `Task` class models a real-time task with:
- Multiple bundles (sequential segments with varying parallelism)
- Period and deadline constraints
- Automatic calculation of total WCET and maximum parallelism

### Processor Class (Lines 31-37)

```python
class Processor:
    """Represents a processing core"""
    def __init__(self, id: int, speed: float = 1.0):
        self.id = id                      # Processor identifier
        self.speed = speed                # Processing speed (for heterogeneous systems)
        self.allocated_tasks = []         # List of tasks assigned to this processor
        self.utilization = 0.0           # Current utilization level
```

Models individual processing cores with varying speeds for heterogeneous systems.

### MultiprocessorSystem Class (Lines 39-53)

```python
class MultiprocessorSystem:
    """Represents the multiprocessor system"""
    def __init__(self, num_cores: int, heterogeneous: bool = True):
        self.num_cores = num_cores
        if heterogeneous:
            # Create processors with varying speeds (0.8 to 1.2)
            self.processors = [Processor(i, 0.8 + 0.4 * random.random()) 
                             for i in range(num_cores)]
        else:
            self.processors = [Processor(i) for i in range(num_cores)]
```

Creates a multiprocessor system with either homogeneous or heterogeneous cores. Heterogeneous cores have speeds between 0.8 and 1.2.

## Task Generation

### TaskGenerator Class (Lines 55-82)

```python
@staticmethod
def generate_tasks(num_tasks: int, target_utilization: float, 
                  num_cores: int) -> List[Task]:
    tasks = []
    
    for i in range(num_tasks):
        # Generate period (100 to 1000 time units)
        period = random.uniform(100, 1000)
        
        # Generate number of bundles (1 to 5)
        num_bundles = random.randint(1, 5)
        
        # Generate bundles
        bundles = []
        remaining_util = target_utilization / num_tasks
        
        for j in range(num_bundles):
            # Execution time based on remaining utilization
            exec_time = random.uniform(0.1, 0.3) * period * remaining_util
            # Required cores (1 to min(4, num_cores))
            req_cores = random.randint(1, min(4, num_cores))
            bundles.append(Bundle(exec_time, req_cores))
```

Generates synthetic tasks with:
- Random periods between 100-1000 time units
- 1-5 bundles per task
- Execution times proportional to target utilization
- Core requirements between 1 and 4

## Response Time Analysis Methods

### ClosedFormAnalysis (Lines 90-117)

```python
def analyze(self, task: Task, tasks: List[Task], 
            system: MultiprocessorSystem) -> float:
    # Initial response time estimate
    R = task.wcet
    converged = False
    iteration = 0
    max_iterations = 100
    
    while not converged and iteration < max_iterations:
        R_new = task.wcet
        
        # Add interference from higher priority tasks
        for other_task in tasks:
            if other_task.priority > task.priority:
                # Calculate interference
                num_instances = np.ceil(R / other_task.period)
                interference = num_instances * other_task.wcet
                
                # Consider parallel execution reduction
                parallel_factor = min(other_task.max_parallelism, 
                                    system.num_cores) / system.num_cores
                R_new += interference * parallel_factor
```

Implements iterative response-time analysis:
1. Starts with task's WCET as initial estimate
2. Iteratively adds interference from higher-priority tasks
3. Accounts for parallel execution by scaling interference
4. Converges when response time stabilizes

### MILPAnalysis (Lines 119-151)

```python
def analyze(self, task: Task, tasks: List[Task], 
            system: MultiprocessorSystem) -> float:
    # Simplified MILP formulation
    # Variables: start times for each bundle
    num_bundles = sum(len(t.bundles) for t in tasks)
    
    # Objective: minimize task completion time
    c = np.zeros(num_bundles)
    task_bundle_idx = sum(len(t.bundles) for t in tasks if t.id < task.id)
    c[task_bundle_idx + len(task.bundles) - 1] = 1
```

Uses Mixed-Integer Linear Programming:
- Variables represent bundle start times
- Objective minimizes target task completion time
- Constraints ensure resource limits are respected

## Allocation Strategies

### DecreasingWorstFit (Lines 158-169)

```python
def allocate(self, tasks: List[Task], system: MultiprocessorSystem):
    system.reset_allocations()
    # Sort tasks by utilization (decreasing)
    sorted_tasks = sorted(tasks, key=lambda t: t.wcet/t.period, reverse=True)
    
    for task in sorted_tasks:
        # Find processor with lowest utilization
        min_proc = min(system.processors, key=lambda p: p.utilization)
        min_proc.allocated_tasks.append(task)
        min_proc.utilization += task.wcet / task.period
```

1. Sorts tasks by utilization in decreasing order
2. Assigns each task to the least loaded processor
3. Balances load across processors

### BestFitDecreasing (Lines 171-184)

```python
def allocate(self, tasks: List[Task], system: MultiprocessorSystem):
    for task in sorted_tasks:
        task_util = task.wcet / task.period
        # Find processor with smallest sufficient capacity
        suitable_procs = [p for p in system.processors 
                        if p.utilization + task_util <= 1.0]
        if suitable_procs:
            best_proc = max(suitable_procs, key=lambda p: p.utilization)
```

Finds the most loaded processor that can still accommodate the task, maximizing processor utilization.

### FirstFitDecreasing (Lines 186-199)

```python
for task in sorted_tasks:
    task_util = task.wcet / task.period
    # Allocate to first processor with sufficient capacity
    for proc in system.processors:
        if proc.utilization + task_util <= 1.0:
            proc.allocated_tasks.append(task)
            proc.utilization += task_util
            break
```

Assigns tasks to the first processor with sufficient capacity, minimizing search time.

### RoundRobin (Lines 201-210)

```python
def allocate(self, tasks: List[Task], system: MultiprocessorSystem):
    for i, task in enumerate(tasks):
        proc = system.processors[i % system.num_cores]
        proc.allocated_tasks.append(task)
        proc.utilization += task.wcet / task.period
```

Distributes tasks evenly across processors in circular order.

## Scheduling Policies

### LDF (Least Deadline First) (Lines 218-223)

```python
def assign_priorities(self, tasks: List[Task]):
    sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
    for i, task in enumerate(sorted_tasks):
        task.priority = len(tasks) - i
```

Assigns higher priorities to tasks with earlier deadlines.

### HEFT (Heterogeneous Earliest Finish Time) (Lines 225-231)

```python
def assign_priorities(self, tasks: List[Task]):
    for task in tasks:
        # Priority based on critical path length
        task.priority = task.wcet / task.period
```

Prioritizes tasks based on their utilization factor.

### FIFO (First In First Out) (Lines 233-238)

```python
def assign_priorities(self, tasks: List[Task]):
    for i, task in enumerate(tasks):
        task.priority = len(tasks) - i
```

Assigns priorities based on arrival order (first task gets highest priority).

## Experiment Runner

### ExperimentRunner Class (Lines 240-394)

The ExperimentRunner orchestrates the entire experimental process:

#### Initialization (Lines 241-243)
```python
def __init__(self):
    self.results = defaultdict(list)  # Stores all experimental results
```

#### Main Experiment Method (Lines 245-303)
```python
def run_experiment(self, num_tasks: int, num_cores: int, 
                  utilization: float, preemptive: bool = True):
    # Generate tasks
    tasks = TaskGenerator.generate_tasks(num_tasks, utilization, num_cores)
    
    # Create system
    system = MultiprocessorSystem(num_cores)
    
    # Test different configurations
    allocation_strategies = {
        'DWF': DecreasingWorstFit(),
        'BFD': BestFitDecreasing(),
        'FFD': FirstFitDecreasing(),
        'RR': RoundRobin()
    }
```

This method:
1. Generates synthetic tasks based on parameters
2. Creates a multiprocessor system
3. Defines all strategies to test

#### Configuration Testing Loop (Lines 271-303)
```python
for alloc_name, alloc_strategy in allocation_strategies.items():
    for sched_name, sched_policy in scheduling_policies.items():
        for analysis_name, analysis_method in analysis_methods.items():
            # Allocate tasks
            alloc_strategy.allocate(tasks, system)
            
            # Assign priorities
            sched_policy.assign_priorities(tasks)
            
            # Analyze response times
            start_time = time.time()
            response_times = []
            schedulable_count = 0
            
            for task in tasks:
                rt = analysis_method.analyze(task, tasks, system)
                response_times.append(rt)
                if rt <= task.deadline:
                    schedulable_count += 1
            
            computation_time = time.time() - start_time
```

For each combination of allocation strategy, scheduling policy, and analysis method:
1. Allocates tasks to processors
2. Assigns task priorities
3. Analyzes response time for each task
4. Counts schedulable tasks (those meeting deadlines)
5. Measures computation time

#### Result Storage (Lines 290-303)
```python
result = {
    'num_tasks': num_tasks,
    'num_cores': num_cores,
    'utilization': utilization,
    'allocation': alloc_name,
    'scheduling': sched_name,
    'analysis': analysis_name,
    'avg_response_time': np.mean(response_times),
    'max_response_time': np.max(response_times),
    'schedulability_ratio': schedulable_count / num_tasks,
    'computation_time': computation_time,
    'system_utilization': np.mean([p.utilization for p in system.processors])
}
```

Stores comprehensive metrics for each experiment.

### Plotting Methods (Lines 305-394)

#### Response Time Comparison Plot (Lines 308-324)
```python
plt.figure(figsize=(12, 8))
for analysis in ['Closed-Form', 'MILP']:
    data = results_df[results_df['analysis'] == analysis]
    plt.plot(data['num_tasks'], data['avg_response_time'], 
            marker='o', label=analysis)
```

Creates a line plot comparing average response times between analysis methods as task count varies.

#### Schedulability vs Core Count Plot (Lines 326-342)
```python
for num_cores in [4, 8, 16, 32]:
    data = results_df[results_df['num_cores'] == num_cores]
    avg_sched = data.groupby('num_tasks')['schedulability_ratio'].mean()
    plt.plot(avg_sched.index, avg_sched.values, 
            marker='o', label=f'{num_cores} cores')
```

Shows how schedulability ratio changes with different core counts.

#### Utilization Impact Plot (Lines 344-360)
```python
for util in [0.25, 0.5, 0.75]:
    data = results_df[results_df['utilization'] == util]
    avg_rt = data.groupby('num_tasks')['avg_response_time'].mean()
    plt.plot(avg_rt.index, avg_rt.values, 
            marker='o', label=f'Utilization {util}')
```

Demonstrates how system utilization affects response times.

#### Computational Overhead Plot (Lines 362-377)
```python
for analysis in ['Closed-Form', 'MILP']:
    data = results_df[results_df['analysis'] == analysis]
    avg_time = data.groupby('num_tasks')['computation_time'].mean()
    plt.plot(avg_time.index, avg_time.values, 
            marker='o', label=analysis)
plt.yscale('log')  # Log scale for better visualization
```

Compares the computational cost of different analysis methods using logarithmic scale.

#### Schedulability Heatmap (Lines 379-393)
```python
pivot_data = results_df.pivot_table(
    values='schedulability_ratio',
    index='utilization',
    columns='num_cores',
    aggfunc='mean'
)

sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn')
```

Creates a 2D heatmap showing schedulability as a function of utilization and core count.

## Main Execution Function (Lines 407-438)

```python
def main():
    """Main function to run all experiments"""
    print("Starting Gang Task Scheduling Analysis...")
    
    runner = ExperimentRunner()
    
    # Test configurations
    task_counts = [10, 50, 100, 200, 400, 600]
    core_counts = [4, 8, 16, 32]
    utilizations = [0.25, 0.5, 0.75]
    
    total_experiments = len(task_counts) * len(core_counts) * len(utilizations)
    experiment_count = 0
    
    for num_tasks in task_counts:
        for num_cores in core_counts:
            for utilization in utilizations:
                experiment_count += 1
                print(f"Running experiment {experiment_count}/{total_experiments}: "
                      f"Tasks={num_tasks}, Cores={num_cores}, Util={utilization}")
                
                runner.run_experiment(num_tasks, num_cores, utilization)
```

Orchestrates the entire experimental campaign:
- Tests 6 different task counts (10 to 600)
- Tests 4 different core counts (4 to 32)
- Tests 3 utilization levels (25%, 50%, 75%)
- Total: 72 base configurations × multiple strategies = hundreds of experiments

## Report Generation Function (Lines 440-492)

```python
def generate_report(results):
    """Generate a summary report of the analysis"""
    with open('gang_scheduling_report.txt', 'w') as f:
        f.write("Gang Task Scheduling Analysis Report\n")
        
        # Calculate summary statistics
        experiments = results['experiments']
        
        # Average schedulability by configuration
        config_stats = defaultdict(list)
        for exp in experiments:
            key = f"{exp['allocation']}-{exp['scheduling']}-{exp['analysis']}"
            config_stats[key].append(exp['schedulability_ratio'])
```

Generates a comprehensive text report including:
1. Summary statistics for all configurations
2. Best performing configuration identification
3. Scalability analysis across different task counts

### Key Report Sections:

#### Configuration Performance Ranking
```python
for config, values in sorted(config_stats.items(), 
                           key=lambda x: np.mean(x[1]), reverse=True)[:10]:
    f.write(f"  {config}: {np.mean(values):.3f}\n")
```

Lists top 10 configurations by average schedulability.

#### Best Configuration Identification
```python
best_config = max(config_stats.items(), key=lambda x: np.mean(x[1]))
f.write(f"Best configuration: {best_config[0]} "
        f"(Avg schedulability: {np.mean(best_config[1]):.3f})\n")
```

Identifies the single best-performing configuration.

#### Scalability Analysis
```python
task_performance = defaultdict(list)
for exp in experiments:
    task_performance[exp['num_tasks']].append(exp['schedulability_ratio'])

for num_tasks in sorted(task_performance.keys()):
    avg_sched = np.mean(task_performance[num_tasks])
    f.write(f"  {num_tasks} tasks: {avg_sched:.3f} avg schedulability\n")
```

Shows how system performance scales with increasing task counts.

## Key Implementation Features

### 1. **Bundled Task Model**
- Tasks consist of multiple bundles with varying parallelism requirements
- More realistic than fixed-parallelism models
- Captures internal task structure

### 2. **Heterogeneous Multiprocessor Support**
- Processors have different speeds (0.8-1.2x)
- Models real-world systems with varying core capabilities

### 3. **Comprehensive Strategy Testing**
- 4 allocation strategies (DWF, BFD, FFD, RR)
- 3 scheduling policies (LDF, HEFT, FIFO)
- 2 analysis methods (Closed-Form, MILP)
- Total: 24 different configurations per scenario

### 4. **Detailed Performance Metrics**
- Response time (average and maximum)
- Schedulability ratio
- System utilization
- Computational overhead

### 5. **Scalability Testing**
- Task counts: 10 to 600
- Core counts: 4 to 32
- Utilization levels: 25% to 75%

### 6. **Visualization Suite**
- Line plots for trends
- Heatmaps for 2D relationships
- Log-scale plots for computational overhead
- Comprehensive visual analysis

## Usage Instructions

1. **Run the main script:**
   ```bash
   python gang_scheduling_analysis.py
   ```

2. **Generated outputs:**
   - `response_time_comparison.png`: Analysis method comparison
   - `schedulability_comparison.png`: Core count impact
   - `utilization_impact.png`: Utilization effects
   - `computational_overhead.png`: Performance overhead
   - `schedulability_heatmap.png`: 2D utilization/cores analysis
   - `allocation_comparison.png`: Allocation strategy comparison
   - `gang_scheduling_report.txt`: Comprehensive text report

3. **Customization:**
   - Modify `task_counts`, `core_counts`, or `utilizations` in main()
   - Add new allocation strategies by extending `AllocationStrategy`
   - Add new scheduling policies by extending `SchedulingPolicy`
   - Implement more sophisticated MILP formulations

## Performance Considerations

1. **MILP Simplification**: The current MILP implementation uses linear programming for speed. For true MILP:
   ```python
   # Add integer constraints
   integrality = [1] * num_variables  # All variables are integers
   ```

2. **Parallel Execution**: For large experiments, consider parallelizing:
   ```python
   from multiprocessing import Pool
   with Pool() as pool:
       results = pool.map(runner.run_experiment, configurations)
   ```

3. **Memory Usage**: For very large task sets (>1000), consider streaming results to disk rather than keeping in memory.

## Theoretical Background

### Response Time Analysis
The closed-form analysis implements the standard response time equation:
```
R_i = C_i + Σ(⌈R_i/T_j⌉ × C_j)
```
Where:
- R_i: Response time of task i
- C_i: WCET of task i
- T_j: Period of higher-priority task j

### Gang Scheduling Challenges
1. **Resource Contention**: Multiple tasks requiring multiple cores
2. **Synchronization**: Bundle execution must be synchronized
3. **Migration Overhead**: Not explicitly modeled but affects performance

### Schedulability Test
A task set is schedulable if:
```
∀i: R_i ≤ D_i
```
Where D_i is the deadline of task i.

## Future Enhancements

1. **Energy Modeling**: Add power consumption calculations
2. **Migration Costs**: Model task migration overhead
3. **Communication Delays**: Account for inter-core communication
4. **Real Workloads**: Import actual task characteristics
5. **Advanced MILP**: Implement full mixed-integer formulation
6. **GPU Support**: Extend to heterogeneous CPU-GPU systems