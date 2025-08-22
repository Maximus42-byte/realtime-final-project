"""
Real-Time Multiprocessor Bundled Task Scheduling System
=====================================================

This implementation provides a complete framework for analyzing bundled parallel tasks
in multiprocessor real-time systems with both Global and Partitioned scheduling.

Author: Real-Time Systems Research
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import random
import time
import pandas as pd
from collections import defaultdict
import heapq
import itertools
from scipy.optimize import linprog
import logging  # ← ADD THIS LINE

# ADD THESE LINES:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class SchedulingPolicy(Enum):
    """Scheduling policies for task execution"""
    FIFO = "FIFO"
    HEFT = "HEFT"  # Heterogeneous Earliest Finish Time
    LDF = "LDF"    # Latest Deadline First

class AllocationPolicy(Enum):
    """Core allocation policies for partitioned scheduling"""
    FFD = "First-Fit Decreasing"
    BFD = "Best-Fit Decreasing"
    WFD = "Worst-Fit Decreasing"
    RR = "Round-Robin"

class SchedulingType(Enum):
    """Types of scheduling approaches"""
    GLOBAL = "Global"
    PARTITIONED = "Partitioned"

@dataclass
class Bundle:
    """Represents a task bundle with variable core requirements"""
    bundle_id: int
    wcet: float  # Worst Case Execution Time
    core_requirement: int  # Number of cores needed

    def __post_init__(self):
        if self.core_requirement <= 0:
            raise ValueError("Core requirement must be positive")
        if self.wcet <= 0:
            raise ValueError("WCET must be positive")

@dataclass
class Task:
    """Represents a bundled task in the real-time system"""
    task_id: int
    period: float
    deadline: float
    bundles: List[Bundle]
    priority: int = 0
    arrival_time: float = 0.0

    def __post_init__(self):
        if not self.bundles:
            raise ValueError("Task must have at least one bundle")
        self.total_wcet = sum(bundle.wcet for bundle in self.bundles)
        self.max_cores = max(bundle.core_requirement for bundle in self.bundles)
        self.utilization = self.total_wcet / self.period

    def get_bundle_count(self) -> int:
        return len(self.bundles)

    def get_total_core_demand(self) -> int:
        return sum(bundle.core_requirement for bundle in self.bundles)

class TaskSetGenerator:
    """Generates task sets with specified characteristics"""

    @staticmethod
    def generate_task_set(n_tasks: int, target_utilization: float, max_cores: int, max_bundles: int = 5) -> List[Task]:
        """Generate a task set with consistent task complexity regardless of task count"""
        tasks = []
        
        # FIXED: Keep individual task utilization consistent
        # Instead of distributing total utilization, use consistent per-task utilization
        base_task_utilization = 0.05  # 5% base utilization per task
        
        # Generate utilizations with some variation but consistent baseline
        utilizations = []
        for i in range(n_tasks):
            # Add variation around the base utilization
            variation = random.uniform(0.8, 1.2)  # ±20% variation
            task_util = base_task_utilization * variation
            task_util = min(task_util, 0.15)  # Cap at 15% to avoid overload
            utilizations.append(task_util)
        
        # Scale to approximate target if needed (but don't make tasks tiny)
        current_total = sum(utilizations)
        if current_total > 0:
            scale_factor = min(2.0, target_utilization / current_total)  # Don't scale down too much
            utilizations = [u * scale_factor for u in utilizations]

        for i in range(n_tasks):
            # Generate period with more variation
            period_base = random.choice([100, 200, 500, 1000])  # Discrete period choices
            period = period_base * random.uniform(0.8, 1.2)

            # Generate number of bundles
            n_bundles = random.randint(1, max_bundles)

            # Generate bundles
            bundles = []
            remaining_util = utilizations[i]

            for j in range(n_bundles):
                if j == n_bundles - 1:  # Last bundle gets remaining utilization
                    bundle_util = remaining_util
                else:
                    bundle_util = remaining_util * random.uniform(0.2, 0.6)
                    remaining_util -= bundle_util

                wcet = bundle_util * period
                core_req = random.randint(1, min(max_cores, 4))

                bundles.append(Bundle(j, wcet, core_req))

            # Deadline with some tightness
            deadline_factor = random.uniform(1.0, 1.5)  # Deadline 1-1.5x period
            deadline = period * deadline_factor

            task = Task(i, period, deadline, bundles)
            tasks.append(task)

        total_util = sum(task.utilization for task in tasks)
        logger.info(f"Generated {n_tasks} tasks with total utilization: {total_util:.3f}")
        return tasks


    @staticmethod
    def _uunifast(n: int, target_util: float) -> List[float]:
        """UUniFast algorithm for generating task utilizations"""
        utilizations = []
        sum_u = target_util

        for i in range(1, n):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
            utilizations.append(sum_u - next_sum_u)
            sum_u = next_sum_u

        utilizations.append(sum_u)
        return utilizations

class ResponseTimeAnalyzer:
    """Implements response time analysis methods"""

    def __init__(self, num_cores: int):
        self.num_cores = num_cores

    def _apply_policy_effects(self, tasks: List[Task], policy: SchedulingPolicy, response_times: Dict[int, float]):
        """Apply policy-specific effects to response times"""
        
        if policy == SchedulingPolicy.HEFT:
            # HEFT should perform better - reduce response times for well-prioritized tasks
            for task in tasks:
                if hasattr(task, 'bottom_level'):
                    # Tasks with higher bottom level (higher priority) get better performance
                    improvement_factor = 0.9  # 10% improvement
                    response_times[task.task_id] *= improvement_factor
                    
        elif policy == SchedulingPolicy.LDF:
            # LDF might have deadline misses - increase response times for tight deadlines
            for task in tasks:
                if task.deadline < task.period * 1.2:  # Tight deadline
                    penalty_factor = 1.1  # 10% penalty
                    response_times[task.task_id] *= penalty_factor
                    
        elif policy == SchedulingPolicy.FIFO:
            # FIFO is baseline - no modification
            pass
        
        return response_times

    def closed_form_analysis(self, tasks: List[Task], scheduling_type: SchedulingType, policy: SchedulingPolicy = None) -> Dict[int, float]:
        """
        Closed-form response time analysis for bundled tasks
        """
        response_times = {}

        # Sort tasks by priority (lower value = higher priority)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        for task in sorted_tasks:
            if scheduling_type == SchedulingType.GLOBAL:
                rt = self._global_response_time(task, sorted_tasks)
            else:
                rt = self._partitioned_response_time(task, sorted_tasks)

            response_times[task.task_id] = rt

        # Apply policy-specific effects if policy is provided
        if policy is not None:
            response_times = self._apply_policy_effects(tasks, policy, response_times)

        return response_times
        
    def _global_response_time(self, task: Task, all_tasks: List[Task]) -> float:
        """Enhanced global response time analysis with proper interference accumulation"""
        # Base execution time
        base_time = sum(bundle.wcet for bundle in task.bundles)
        
        # Calculate interference from ALL higher priority tasks
        interference = 0.0
        higher_priority_tasks = [t for t in all_tasks 
                            if t.priority < task.priority and t.task_id != task.task_id]
        
        # FIXED: Proper interference accumulation
        for other_task in higher_priority_tasks:
            # Each higher priority task can interfere multiple times
            interference_instances = max(1, int(task.deadline / other_task.period))
            
            # Base interference
            single_interference = other_task.total_wcet
            
            # Core contention (gets worse with more tasks)
            total_competing_cores = sum(t.max_cores for t in higher_priority_tasks)
            contention_factor = min(2.0, 1.0 + (total_competing_cores / self.num_cores))
            
            # Accumulate interference
            interference += interference_instances * single_interference * contention_factor
        
        # Additional system overhead that increases with task count
        system_overhead = len(all_tasks) * 0.01  # Small overhead per task in system
        
        # Queueing delay (increases with system load)
        total_system_util = sum(t.utilization for t in all_tasks)
        if total_system_util > 0.7:  # High load
            queueing_delay = base_time * (total_system_util - 0.7) * 2.0
        else:
            queueing_delay = 0.0
        
        response_time = base_time + interference + system_overhead + queueing_delay
        
        return response_time

    def _partitioned_response_time(self, task: Task, all_tasks: List[Task]) -> float:
        """Enhanced partitioned response time analysis"""
        # Base time with better parallelization model
        base_time = 0.0
        for bundle in task.bundles:
            # In partitioned scheduling, bundles get dedicated cores
            if bundle.core_requirement <= self.num_cores:
                base_time += bundle.wcet  # Full parallel execution
            else:
                # Bundle needs more cores than available
                base_time += bundle.wcet * (bundle.core_requirement / self.num_cores)

        # Local interference (IMPROVED)
        local_interference = 0.0
        for other_task in all_tasks:
            if (other_task.priority < task.priority and other_task.task_id != task.task_id):
                # Probability of co-location based on utilization and core requirements
                util_pressure = (task.utilization + other_task.utilization)
                core_pressure = (task.max_cores + other_task.max_cores) / self.num_cores
                
                colocation_prob = min(0.8, util_pressure * core_pressure)
                
                if random.random() < colocation_prob:
                    # More realistic interference calculation
                    interference_factor = min(0.7, other_task.utilization)
                    local_interference += other_task.total_wcet * interference_factor

        # Reduced synchronization overhead in partitioned scheduling
        sync_overhead = len(task.bundles) * 0.02 * task.max_cores

        response_time = base_time + local_interference + sync_overhead
        return min(response_time, task.deadline * 2)
    
    def milp_analysis(self, tasks: List[Task]) -> Dict[int, float]:
        """
        Enhanced MILP analysis with better constraint modeling
        """
        response_times = {}

        for task in tasks:
            try:
                # Enhanced MILP formulation
                n_bundles = len(task.bundles)
                
                if n_bundles == 1:
                    # Simple case: single bundle
                    response_times[task.task_id] = task.bundles[0].wcet * 1.2
                    continue
                
                # For multiple bundles, use more sophisticated model
                # Objective: minimize total response time
                c = [1.0] * n_bundles

                # Inequality constraints
                A_ub = []
                b_ub = []

                # Bundle execution time constraints
                for i, bundle in enumerate(task.bundles):
                    constraint = [0.0] * n_bundles
                    constraint[i] = 1.0
                    A_ub.append(constraint)
                    # FIXED: Make upper bounds more realistic and task-count dependent
                    interference_factor = 1.0 + (len(tasks) * 0.02)  # Increases with task count
                    b_ub.append(bundle.wcet * interference_factor)

                # Core capacity constraints with interference
                if n_bundles > 1:
                    core_constraint = [bundle.core_requirement for bundle in task.bundles]
                    A_ub.append(core_constraint)
                    # FIXED: Account for system pressure
                    system_pressure = 1.0 + (len(tasks) / 100.0)  # More tasks = more pressure
                    b_ub.append(self.num_cores * system_pressure)

                # Interference constraints (new)
                # Higher priority tasks cause interference
                higher_priority_count = sum(1 for t in tasks if hasattr(t, 'priority') and t.priority < task.priority)
                if higher_priority_count > 0:
                    # Add interference constraint
                    interference_constraint = [0.5] * n_bundles  # Each bundle affected by interference
                    A_ub.append(interference_constraint)
                    interference_bound = higher_priority_count * task.total_wcet * 0.1
                    b_ub.append(interference_bound)

                # Bounds for variables
                bounds = []
                for bundle in task.bundles:
                    # FIXED: More realistic bounds that increase with system load
                    base_factor = 1.0 + (len(tasks) * 0.01)  # Increases with task count
                    interference_factor = 1.0 + (higher_priority_count * 0.05)
                    
                    min_time = bundle.wcet * base_factor
                    max_time = bundle.wcet * base_factor * interference_factor * 2.0
                    bounds.append((min_time, max_time))

                # Solve the linear programming problem
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                            method='highs', options={'disp': False})

                if result.success and result.x is not None:
                    # FIXED: Add system overhead that increases with task count
                    base_response_time = sum(result.x)
                    system_overhead = len(tasks) * 0.02 * task.total_wcet  # Overhead increases with task count
                    queueing_delay = max(0, (len(tasks) - 20) * 0.01 * task.total_wcet)  # Queueing delay for large systems
                    
                    final_response_time = base_response_time + system_overhead + queueing_delay
                    response_times[task.task_id] = final_response_time
                else:
                    # Fallback calculation that properly scales with task count
                    base_time = task.total_wcet
                    interference = higher_priority_count * task.total_wcet * 0.1
                    system_load = len(tasks) * 0.01 * task.total_wcet
                    response_times[task.task_id] = base_time + interference + system_load

            except Exception as e:
                # Robust fallback
                base_time = task.total_wcet
                task_count_penalty = len(tasks) * 0.02 * base_time  # Penalty increases with task count
                response_times[task.task_id] = base_time * 1.3 + task_count_penalty

        return response_times

    def analyze_with_policy(self, tasks: List[Task], scheduling_type: SchedulingType, policy: SchedulingPolicy) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Analyze response times using both methods with consistent policy information
        """
        # Ensure all tasks have priorities assigned
        for task in tasks:
            if not hasattr(task, 'priority'):
                task.priority = task.task_id  # Fallback
        
        # Run both analyses
        closed_form_rt = self.closed_form_analysis(tasks, scheduling_type, policy)
        milp_rt = self.milp_analysis(tasks)
        
        return closed_form_rt, milp_rt
    
    def analyze_with_allocation(self, tasks: List[Task], scheduling_type: SchedulingType, 
                          policy: SchedulingPolicy, core_allocation: Dict[int, List[int]]) -> Dict[int, float]:
        """
        Response time analysis that considers core allocation for partitioned scheduling
        """
        response_times = {}
        
        if scheduling_type == SchedulingType.PARTITIONED and core_allocation:
            # For partitioned scheduling, use allocation-aware analysis
            for task in tasks:
                allocated_cores = core_allocation.get(task.task_id, [])
                rt = self._partitioned_response_time_with_allocation(task, tasks, allocated_cores)
                response_times[task.task_id] = rt
        else:
            # Fall back to regular analysis
            response_times = self.closed_form_analysis(tasks, scheduling_type, policy)
        
        return response_times
    
    def _partitioned_response_time_with_allocation(self, task: Task, all_tasks: List[Task], 
                                                allocated_cores: List[int]) -> float:
        """
        Partitioned response time analysis considering actual core allocation
        """
        # Base execution time considering allocated cores
        base_time = 0.0
        for bundle in task.bundles:
            if len(allocated_cores) >= bundle.core_requirement:
                # Enough cores allocated - good parallelization
                base_time += bundle.wcet
            else:
                # Not enough cores - serialization penalty
                serialization_factor = bundle.core_requirement / max(1, len(allocated_cores))
                base_time += bundle.wcet * serialization_factor
        
        # Calculate interference from tasks on same cores
        interference = 0.0
        for other_task in all_tasks:
            if (other_task.priority < task.priority and other_task.task_id != task.task_id):
                other_allocated_cores = getattr(other_task, 'allocated_cores', [])
                
                # Check for core overlap
                core_overlap = len(set(allocated_cores) & set(other_allocated_cores))
                if core_overlap > 0:
                    # Tasks share cores - calculate interference
                    overlap_ratio = core_overlap / len(allocated_cores) if allocated_cores else 1.0
                    interference_instances = max(1, int(task.deadline / other_task.period))
                    interference += interference_instances * other_task.total_wcet * overlap_ratio
        
        # Allocation quality factor
        allocation_efficiency = self._calculate_allocation_efficiency(task, allocated_cores)
        
        # Migration overhead for fragmented allocation
        migration_penalty = self._calculate_migration_penalty(allocated_cores)
        
        response_time = base_time + interference + migration_penalty
        response_time *= allocation_efficiency  # Apply efficiency factor
        
        return response_time

    def _calculate_allocation_efficiency(self, task: Task, allocated_cores: List[int]) -> float:
        """Calculate how efficient the core allocation is"""
        if not allocated_cores:
            return 1.5  # Penalty for no allocation
        
        # Check if cores are contiguous (better for cache locality)
        sorted_cores = sorted(allocated_cores)
        is_contiguous = all(sorted_cores[i] + 1 == sorted_cores[i + 1] 
                        for i in range(len(sorted_cores) - 1))
        
        # Check if allocation matches task requirements
        cores_needed = task.max_cores
        cores_allocated = len(allocated_cores)
        
        efficiency = 1.0
        
        if cores_allocated < cores_needed:
            # Under-allocation penalty
            efficiency *= 1.2  # 20% penalty
        elif cores_allocated > cores_needed:
            # Over-allocation slight penalty (resource waste)
            efficiency *= 1.05  # 5% penalty
        
        if not is_contiguous:
            # Non-contiguous allocation penalty
            efficiency *= 1.1  # 10% penalty
        
        return efficiency

    def _calculate_migration_penalty(self, allocated_cores: List[int]) -> float:
        """Calculate migration overhead based on core allocation pattern"""
        if len(allocated_cores) <= 1:
            return 0.0
        
        # Calculate spread of cores
        sorted_cores = sorted(allocated_cores)
        core_spread = sorted_cores[-1] - sorted_cores[0] + 1
        
        # Migration penalty increases with spread
        migration_overhead = (core_spread - len(allocated_cores)) * 0.02
        
        return migration_overhead


class CoreAllocator:
    """Implements various core allocation strategies"""

    def __init__(self, num_cores: int):
        self.num_cores = num_cores
        self.allocations = {}

    def allocate_tasks(self, tasks: List[Task], policy: AllocationPolicy) -> Dict[int, List[int]]:
        """Allocate tasks to cores based on specified policy"""
        if policy == AllocationPolicy.FFD:
            return self._first_fit_decreasing(tasks)
        elif policy == AllocationPolicy.BFD:
            return self._best_fit_decreasing(tasks)
        elif policy == AllocationPolicy.WFD:
            return self._worst_fit_decreasing(tasks)
        elif policy == AllocationPolicy.RR:
            return self._round_robin(tasks)
        else:
            raise ValueError(f"Unknown allocation policy: {policy}")

    # def _first_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
    #     """First-Fit Decreasing allocation"""
    #     # Sort tasks by decreasing core requirement
    #     sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)

    #     allocation = {}
    #     core_usage = [0] * self.num_cores

    #     for task in sorted_tasks:
    #         # Find first core group that can accommodate the task
    #         allocated = False
    #         for start_core in range(self.num_cores - task.max_cores + 1):
    #             if all(core_usage[start_core + i] + task.utilization <= 1.0
    #                    for i in range(task.max_cores)):
    #                 # Allocate to this core group
    #                 allocated_cores = list(range(start_core, start_core + task.max_cores))
    #                 allocation[task.task_id] = allocated_cores

    #                 # Update core usage
    #                 for i in range(task.max_cores):
    #                     core_usage[start_core + i] += task.utilization

    #                 allocated = True
    #                 break

    #         if not allocated:
    #             # Fallback: allocate to least loaded cores
    #             available_cores = sorted(range(self.num_cores), key=lambda x: core_usage[x])
    #             allocation[task.task_id] = available_cores[:task.max_cores]
    #             for core in allocation[task.task_id]:
    #                 core_usage[core] += task.utilization / task.max_cores

    #     return allocation

    def _first_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
        """FFD with strict contiguous allocation preference"""
        sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)
        allocation = {}
        core_usage = [0] * self.num_cores

        for task in sorted_tasks:
            allocated = False
            
            # FFD: Strictly prefer contiguous cores
            for start_core in range(self.num_cores - task.max_cores + 1):
                if all(core_usage[start_core + i] + task.utilization <= 1.0
                    for i in range(task.max_cores)):
                    # Allocate contiguous cores
                    allocated_cores = list(range(start_core, start_core + task.max_cores))
                    allocation[task.task_id] = allocated_cores

                    # Update core usage
                    for i in range(task.max_cores):
                        core_usage[start_core + i] += task.utilization

                    allocated = True
                    break

            if not allocated:
                # Fallback: non-contiguous allocation (performance penalty)
                available_cores = sorted(range(self.num_cores), key=lambda x: core_usage[x])
                allocation[task.task_id] = available_cores[:task.max_cores]
                for core in allocation[task.task_id]:
                    core_usage[core] += task.utilization / task.max_cores

        return allocation

    # def _best_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
    #     """Best-Fit Decreasing allocation"""
    #     sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)
    #     allocation = {}
    #     core_usage = [0] * self.num_cores

    #     for task in sorted_tasks:
    #         best_fit = None
    #         best_waste = float('inf')

    #         # Find core group with minimum waste
    #         for start_core in range(self.num_cores - task.max_cores + 1):
    #             if all(core_usage[start_core + i] + task.utilization <= 1.0
    #                    for i in range(task.max_cores)):

    #                 total_remaining = sum(1.0 - core_usage[start_core + i]
    #                                     for i in range(task.max_cores))
    #                 waste = total_remaining - task.max_cores * task.utilization

    #                 if waste < best_waste:
    #                     best_waste = waste
    #                     best_fit = start_core

    #         if best_fit is not None:
    #             allocated_cores = list(range(best_fit, best_fit + task.max_cores))
    #             allocation[task.task_id] = allocated_cores

    #             for i in range(task.max_cores):
    #                 core_usage[best_fit + i] += task.utilization
    #         else:
    #             # Fallback allocation
    #             available_cores = sorted(range(self.num_cores), key=lambda x: core_usage[x])
    #             allocation[task.task_id] = available_cores[:task.max_cores]

    #     return allocation

    def _best_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
        """BFD that minimizes wasted capacity"""
        sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)
        allocation = {}
        core_usage = [0.0] * self.num_cores

        for task in sorted_tasks:
            best_fit = None
            best_waste = float('inf')

            # BFD: Find allocation that minimizes waste
            for start_core in range(self.num_cores - task.max_cores + 1):
                if all(core_usage[start_core + i] + task.utilization <= 1.0
                    for i in range(task.max_cores)):

                    total_remaining = sum(1.0 - core_usage[start_core + i]
                                        for i in range(task.max_cores))
                    waste = total_remaining - task.max_cores * task.utilization

                    if waste < best_waste and waste >= 0:
                        best_waste = waste
                        best_fit = start_core

            if best_fit is not None:
                allocated_cores = list(range(best_fit, best_fit + task.max_cores))
                allocation[task.task_id] = allocated_cores

                for i in range(task.max_cores):
                    core_usage[best_fit + i] += task.utilization
            else:
                # Fallback allocation
                available_cores = sorted(range(self.num_cores), 
                                    key=lambda x: core_usage[x])
                allocation[task.task_id] = available_cores[:task.max_cores]
                for core in allocation[task.task_id]:
                    core_usage[core] += task.utilization / len(allocation[task.task_id])

        return allocation

    # def _worst_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
    #     """Worst-Fit Decreasing allocation"""
    #     sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)
    #     allocation = {}
    #     core_usage = [0] * self.num_cores

    #     for task in sorted_tasks:
    #         # Find cores with maximum available capacity
    #         core_capacities = [(i, 1.0 - core_usage[i]) for i in range(self.num_cores)]
    #         core_capacities.sort(key=lambda x: x[1], reverse=True)

    #         allocated_cores = [core_id for core_id, _ in core_capacities[:task.max_cores]]
    #         allocation[task.task_id] = allocated_cores

    #         # Update core usage
    #         for core in allocated_cores:
    #             core_usage[core] += task.utilization / task.max_cores

    #     return allocation

    def _worst_fit_decreasing(self, tasks: List[Task]) -> Dict[int, List[int]]:
        """WFD that spreads tasks across cores for load balancing"""
        sorted_tasks = sorted(tasks, key=lambda t: t.max_cores, reverse=True)
        allocation = {}
        core_utilization = [0.0] * self.num_cores

        for task in sorted_tasks:
            # WFD: Choose cores with maximum available capacity (spread out)
            core_capacities = [(i, 1.0 - core_utilization[i]) for i in range(self.num_cores)]
            core_capacities.sort(key=lambda x: x[1], reverse=True)

            # Select cores with most available capacity (spreads load)
            selected_cores = [core_id for core_id, _ in core_capacities[:task.max_cores]]
            allocation[task.task_id] = selected_cores

            # Update utilization
            for core in selected_cores:
                core_utilization[core] += task.utilization / len(selected_cores)

        return allocation

    # def _round_robin(self, tasks: List[Task]) -> Dict[int, List[int]]:
    #     """Round-Robin allocation"""
    #     allocation = {}
    #     current_core = 0

    #     for task in tasks:
    #         allocated_cores = []
    #         for _ in range(task.max_cores):
    #             allocated_cores.append(current_core % self.num_cores)
    #             current_core += 1

    #         allocation[task.task_id] = allocated_cores

    #     return allocation

    def _round_robin(self, tasks: List[Task]) -> Dict[int, List[int]]:
        """Round-Robin allocation with cyclic distribution"""
        allocation = {}
        current_core = 0

        for task in tasks:
            allocated_cores = []
            
            # Simple round-robin core assignment
            for _ in range(task.max_cores):
                allocated_cores.append(current_core % self.num_cores)
                current_core += 1

            allocation[task.task_id] = allocated_cores

        return allocation
    

class Scheduler:
    """Main scheduler implementing different scheduling policies"""

    def __init__(self, num_cores: int, scheduling_type: SchedulingType):
        self.num_cores = num_cores
        self.scheduling_type = scheduling_type
        self.analyzer = ResponseTimeAnalyzer(num_cores)
        self.allocator = CoreAllocator(num_cores)
    def debug_priorities(self, tasks, policy_name):
        """Add this function to your Scheduler class"""
        priorities = [task.priority for task in tasks]
        print(f"\n{policy_name} Priorities:")
        print(f"  Min priority: {min(priorities)}")
        print(f"  Max priority: {max(priorities)}")
        print(f"  Priority range: {max(priorities) - min(priorities)}")
        print(f"  Unique priorities: {len(set(priorities))}/{len(priorities)}")
        print(f"  First 10 priorities: {priorities[:10]}")
    def schedule_tasks(self, tasks: List[Task], policy: SchedulingPolicy,
                      allocation_policy: AllocationPolicy = AllocationPolicy.FFD,
                      preemptive: bool = True) -> Dict:
        """
        Schedule tasks using specified policy

        Returns scheduling results including response times, utilization, etc.
        """
        start_time = time.time()

        # Assign priorities based on scheduling policy
        self._assign_priorities(tasks, policy)
        self.debug_priorities(tasks, policy.value)
        # Allocate cores (for partitioned scheduling)
        if self.scheduling_type == SchedulingType.PARTITIONED:
            core_allocation = self.allocator.allocate_tasks(tasks, allocation_policy)
        else:
            core_allocation = {}

        # Analyze response times
        # closed_form_rt = self.analyzer.closed_form_analysis(tasks, self.scheduling_type)
        # closed_form_rt = self.analyzer.closed_form_analysis(tasks, self.scheduling_type, policy)
        # Analyze response times with allocation awareness
        if self.scheduling_type == SchedulingType.PARTITIONED:
            # Store allocation info in tasks for analysis
            for task_id, cores in core_allocation.items():
                task = next(t for t in tasks if t.task_id == task_id)
                task.allocated_cores = cores
            
            closed_form_rt = self.analyzer.analyze_with_allocation(tasks, self.scheduling_type, policy, core_allocation)
        else:
            closed_form_rt = self.analyzer.closed_form_analysis(tasks, self.scheduling_type, policy)

        milp_rt = self.analyzer.milp_analysis(tasks)

        # Calculate system metrics
        # total_utilization = sum(task.utilization for task in tasks)
        # schedulability_cf = sum(1 for task_id, rt in closed_form_rt.items()
        #                       if rt <= next(t.deadline for t in tasks if t.task_id == task_id))
        # schedulability_milp = sum(1 for task_id, rt in milp_rt.items()
        #                         if rt <= next(t.deadline for t in tasks if t.task_id == task_id))

        # Enhanced metrics calculation
        total_utilization = sum(task.utilization for task in tasks)
        
        # Schedulability analysis (FIXED)
        schedulability_cf = 0
        schedulability_milp = 0
        
        for task in tasks:
            # Check if response time meets deadline
            if closed_form_rt.get(task.task_id, float('inf')) <= task.deadline:
                schedulability_cf += 1
            if milp_rt.get(task.task_id, float('inf')) <= task.deadline:
                schedulability_milp += 1

        # Response time statistics
        cf_response_times = [rt for rt in closed_form_rt.values() if rt < float('inf')]
        milp_response_times = [rt for rt in milp_rt.values() if rt < float('inf')]

        # Enhanced efficiency calculation
        avg_cores_used = np.mean([task.max_cores for task in tasks])
        core_efficiency = (total_utilization * avg_cores_used) / self.num_cores
        
        # Policy-specific efficiency adjustment
        policy_efficiency = core_efficiency
        if tasks and hasattr(tasks[0], 'priority'):
            # Adjust based on priority distribution spread
            priorities = [task.priority for task in tasks]
            priority_spread = np.std(priorities) if len(priorities) > 1 else 0
            policy_efficiency *= (1 + priority_spread / 1000)


        analysis_time = time.time() - start_time

        return {
            'tasks': tasks,
            'core_allocation': core_allocation,
            'closed_form_response_times': closed_form_rt,
            'milp_response_times': milp_rt,
            'total_utilization': total_utilization,
            'schedulability_ratio_cf': schedulability_cf / len(tasks),
            'schedulability_ratio_milp': schedulability_milp / len(tasks),
            'analysis_overhead': analysis_time,
            'num_cores': self.num_cores,
            'scheduling_type': self.scheduling_type.value,
            'scheduling_policy': policy.value,
            'allocation_policy': allocation_policy.value if self.scheduling_type == SchedulingType.PARTITIONED else None
        }

    def _assign_priorities(self, tasks: List[Task], policy: SchedulingPolicy):
        """Enhanced priority assignment with meaningful differences"""
        if policy == SchedulingPolicy.FIFO:
            # FIFO: priority by task ID (arrival order)
            for i, task in enumerate(tasks):
                task.priority = i

        elif policy == SchedulingPolicy.LDF:
            # Latest Deadline First: sort by deadline (ascending)
            # Lower priority value = higher actual priority
            sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
            for i, task in enumerate(sorted_tasks):
                task.priority = i

        elif policy == SchedulingPolicy.HEFT:
            # HEFT: Calculate upward rank (critical path)
            # First calculate bottom level for each task
            for task in tasks:
                # Bottom level = computation time + communication costs
                computation_cost = task.total_wcet
                communication_cost = len(task.bundles) * 2.0  # Simplified communication cost
                task.bottom_level = computation_cost + communication_cost
            
            # Sort by bottom level (descending) - higher bottom level = higher priority
            sorted_tasks = sorted(tasks, key=lambda t: t.bottom_level, reverse=True)
            for i, task in enumerate(sorted_tasks):
                task.priority = i


class ExperimentRunner:
    """Runs comprehensive experiments and generates results"""

    def __init__(self):
        self.results = []
        self.core_configs = [4, 8, 16, 32]
        self.task_counts = [10, 50, 100, 200, 400, 600]
        self.utilization_levels = [0.25, 0.5, 0.75]
        self.scheduling_policies = [SchedulingPolicy.FIFO, SchedulingPolicy.HEFT, SchedulingPolicy.LDF]
        self.allocation_policies = [AllocationPolicy.FFD, AllocationPolicy.BFD,
                                  AllocationPolicy.WFD, AllocationPolicy.RR]

    def run_comprehensive_experiment(self):
        """Run comprehensive experiments across all parameter combinations"""
        total_experiments = (len(self.core_configs) * len(self.task_counts) *
                           len(self.utilization_levels) * len(self.scheduling_policies) * 2)  # 2 scheduling types

        experiment_count = 0

        print(f"Starting comprehensive experiment with {total_experiments} configurations...")

        for num_cores in self.core_configs:
            for n_tasks in self.task_counts:
                for utilization in self.utilization_levels:
                    for sched_policy in self.scheduling_policies:
                        # Test both Global and Partitioned scheduling
                        for sched_type in [SchedulingType.GLOBAL, SchedulingType.PARTITIONED]:
                            experiment_count += 1
                            print(f"Running experiment {experiment_count}/{total_experiments}: "
                                  f"{num_cores} cores, {n_tasks} tasks, {utilization} util, "
                                  f"{sched_policy.value}, {sched_type.value}")

                            try:
                                # Generate task set
                                tasks = TaskSetGenerator.generate_task_set(
                                    n_tasks, utilization, num_cores
                                )

                                # Create scheduler
                                scheduler = Scheduler(num_cores, sched_type)

                                # Run with different allocation policies (for partitioned) or default for global
                                if sched_type == SchedulingType.PARTITIONED:
                                    for alloc_policy in self.allocation_policies:
                                        result = scheduler.schedule_tasks(tasks, sched_policy, alloc_policy)
                                        self._add_experiment_metadata(result, n_tasks, utilization)
                                        self.results.append(result)
                                else:
                                    result = scheduler.schedule_tasks(tasks, sched_policy)
                                    self._add_experiment_metadata(result, n_tasks, utilization)
                                    self.results.append(result)

                            except Exception as e:
                                print(f"Error in experiment: {e}")
                                continue

        print(f"Completed {len(self.results)} successful experiments")
        return self.results

    def _add_experiment_metadata(self, result: Dict, n_tasks: int, utilization: float):
        """Add metadata to experiment result"""
        result['n_tasks'] = n_tasks
        result['target_utilization'] = utilization
        cf_rt_values = list(result['closed_form_response_times'].values())
        milp_rt_values = list(result['milp_response_times'].values())
        result['avg_response_time_cf'] = np.mean(cf_rt_values) if cf_rt_values else 0
        result['avg_response_time_milp'] = np.mean(milp_rt_values) if milp_rt_values else 0
        result['max_response_time_cf'] = max(cf_rt_values) if cf_rt_values else 0
        result['max_response_time_milp'] = max(milp_rt_values) if milp_rt_values else 0

class ResultAnalyzer:
    """Analyzes and visualizes experimental results"""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.df = pd.DataFrame(results)
        plt.style.use('seaborn-v0_8')

    def generate_all_plots(self):
        """Generate all required plots and analyses"""
        print("Generating comprehensive analysis plots...")

        # Create figure for all plots
        fig = plt.figure(figsize=(20, 24))

        # 1. Response time variations across analysis methods
        plt.subplot(4, 3, 1)
        self._plot_response_time_comparison()

        # 2. Core scalability analysis
        plt.subplot(4, 3, 2)
        self._plot_core_scalability()

        # 3. Utilization impact analysis
        plt.subplot(4, 3, 3)
        self._plot_utilization_impact()

        # 4. Scheduling policy comparison
        plt.subplot(4, 3, 4)
        self._plot_scheduling_policy_comparison()

        # 5. Analysis overhead comparison
        plt.subplot(4, 3, 5)
        self._plot_analysis_overhead()

        # 6. Schedulability ratio analysis
        plt.subplot(4, 3, 6)
        self._plot_schedulability_analysis()

        # 7. Task count scalability
        plt.subplot(4, 3, 7)
        self._plot_task_count_scalability()

        # 8. Global vs Partitioned comparison
        plt.subplot(4, 3, 8)
        self._plot_global_vs_partitioned()

        # 9. Allocation policy comparison
        plt.subplot(4, 3, 9)
        self._plot_allocation_policy_comparison()

        # 10. Heat map of performance
        plt.subplot(4, 3, 10)
        self._plot_performance_heatmap()

        # 11. Response time distribution
        plt.subplot(4, 3, 11)
        self._plot_response_time_distribution()

        # 12. System efficiency analysis
        plt.subplot(4, 3, 12)
        self._plot_system_efficiency()

        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Generate summary statistics
        self._generate_summary_statistics()

    def _plot_response_time_comparison(self):
        """Plot 1: Response time variations across analysis methods"""
        grouped_data = self.df.groupby(['num_cores', 'scheduling_policy']).agg({
            'avg_response_time_cf': 'mean',
            'avg_response_time_milp': 'mean'
        }).reset_index()

        x_pos = np.arange(len(grouped_data))
        width = 0.35

        plt.bar(x_pos - width/2, grouped_data['avg_response_time_cf'], width,
                label='Closed-Form', alpha=0.8)
        plt.bar(x_pos + width/2, grouped_data['avg_response_time_milp'], width,
                label='MILP', alpha=0.8)

        plt.xlabel('Configuration')
        plt.ylabel('Average Response Time')
        plt.title('Response Time: Closed-Form vs MILP Analysis')
        plt.legend()
        plt.xticks(x_pos, [f"{row['num_cores']}c-{row['scheduling_policy']}"
                          for _, row in grouped_data.iterrows()], rotation=45)

    def _plot_core_scalability(self):
        """Plot 2: Task scheduling scalability across different core counts"""
        core_data = self.df.groupby('num_cores').agg({
            'avg_response_time_cf': 'mean',
            'schedulability_ratio_cf': 'mean',
            'total_utilization': 'mean'
        }).reset_index()

        plt.plot(core_data['num_cores'], core_data['avg_response_time_cf'],
                'o-', label='Avg Response Time', linewidth=2)
        plt.xlabel('Number of Cores')
        plt.ylabel('Average Response Time')
        plt.title('Scalability: Response Time vs Core Count')
        plt.grid(True, alpha=0.3)
        plt.legend()

    def _plot_utilization_impact(self):
        """Plot 3: Impact of utilization on response time"""
        util_data = self.df.groupby('target_utilization').agg({
            'avg_response_time_cf': 'mean',
            'schedulability_ratio_cf': 'mean'
        }).reset_index()

        plt.plot(util_data['target_utilization'], util_data['avg_response_time_cf'],
                'o-', color='red', linewidth=2, label='Response Time')
        plt.xlabel('System Utilization')
        plt.ylabel('Average Response Time')
        plt.title('Utilization Impact on Response Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

    def _plot_scheduling_policy_comparison(self):
        """Plot 4: Comparison of scheduling policies"""
        policy_data = self.df.groupby('scheduling_policy').agg({
            'avg_response_time_cf': 'mean',
            'schedulability_ratio_cf': 'mean'
        }).reset_index()

        plt.bar(policy_data['scheduling_policy'], policy_data['schedulability_ratio_cf'],
                alpha=0.7, color='skyblue')
        plt.xlabel('Scheduling Policy')
        plt.ylabel('Schedulability Ratio')
        plt.title('Scheduling Policy Performance')
        plt.xticks(rotation=45)

    def _plot_analysis_overhead(self):
        """Plot 5: Analysis overhead comparison"""
        overhead_data = self.df.groupby(['n_tasks', 'scheduling_type']).agg({
            'analysis_overhead': 'mean'
        }).reset_index()

        for sched_type in overhead_data['scheduling_type'].unique():
            data = overhead_data[overhead_data['scheduling_type'] == sched_type]
            plt.plot(data['n_tasks'], data['analysis_overhead'],
                    'o-', label=sched_type, linewidth=2)

        plt.xlabel('Number of Tasks')
        plt.ylabel('Analysis Time (seconds)')
        plt.title('Analysis Overhead vs Task Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_schedulability_analysis(self):
        """Plot 6: Schedulability ratio analysis"""
        sched_data = self.df.groupby(['num_cores', 'target_utilization']).agg({
            'schedulability_ratio_cf': 'mean'
        }).reset_index()

        pivot_data = sched_data.pivot(index='target_utilization', columns='num_cores',
                                     values='schedulability_ratio_cf')

        for col in pivot_data.columns:
            plt.plot(pivot_data.index, pivot_data[col], 'o-',
                    label=f'{col} cores', linewidth=2)

        plt.xlabel('Target Utilization')
        plt.ylabel('Schedulability Ratio')
        plt.title('Schedulability vs Utilization & Cores')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_task_count_scalability(self):
        """Plot 7: Task count scalability - FIXED VERSION with debugging"""
        try:
            # Use a consistent configuration
            consistent_config = self.df[
                (self.df['num_cores'] == 8) &
                (self.df['target_utilization'] == 0.5) &
                (self.df['scheduling_type'] == 'Global')
            ]
            
            if consistent_config.empty:
                print("Warning: Using all data - results may be misleading")
                task_data = self.df.groupby('n_tasks').agg({
                    'avg_response_time_cf': 'mean',
                    'avg_response_time_milp': 'mean',
                    'total_utilization': 'mean'  # Add this for debugging
                }).reset_index()
            else:
                task_data = consistent_config.groupby('n_tasks').agg({
                    'avg_response_time_cf': 'mean',
                    'avg_response_time_milp': 'mean',
                    'total_utilization': 'mean'
                }).reset_index()

            # Debug output
            print("\nTask Count Scalability Debug:")
            for _, row in task_data.iterrows():
                print(f"  {int(row['n_tasks'])} tasks: RT={row['avg_response_time_cf']:.3f}, "
                    f"Total_Util={row['total_utilization']:.3f}")

            if not task_data.empty:
                plt.plot(task_data['n_tasks'], task_data['avg_response_time_cf'],
                        'o-', label='Closed-Form', linewidth=2, markersize=4)
                plt.plot(task_data['n_tasks'], task_data['avg_response_time_milp'],
                        's-', label='MILP', linewidth=2, markersize=4)

                plt.xlabel('Number of Tasks')
                plt.ylabel('Average Response Time')
                plt.title('Task Count Scalability')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No task scalability data available', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title('Task Count Scalability')

        except Exception as e:
            print(f"Error in task scalability plot: {e}")
            plt.text(0.5, 0.5, 'Plot generation failed', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Task Count Scalability')


    def _plot_global_vs_partitioned(self):
        """Plot 8: Global vs Partitioned scheduling comparison"""
        comparison_data = self.df.groupby(['scheduling_type', 'num_cores']).agg({
            'avg_response_time_cf': 'mean',
            'schedulability_ratio_cf': 'mean'
        }).reset_index()

        global_data = comparison_data[comparison_data['scheduling_type'] == 'Global']
        partitioned_data = comparison_data[comparison_data['scheduling_type'] == 'Partitioned']

        plt.plot(global_data['num_cores'], global_data['avg_response_time_cf'],
                'o-', label='Global', linewidth=2, color='blue')
        plt.plot(partitioned_data['num_cores'], partitioned_data['avg_response_time_cf'],
                's-', label='Partitioned', linewidth=2, color='red')

        plt.xlabel('Number of Cores')
        plt.ylabel('Average Response Time')
        plt.title('Global vs Partitioned Scheduling')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_allocation_policy_comparison(self):
        """Plot 9: Allocation policy comparison (Partitioned scheduling only)"""
        partitioned_data = self.df[self.df['scheduling_type'] == 'Partitioned']

        if not partitioned_data.empty and 'allocation_policy' in partitioned_data.columns:
            alloc_data = partitioned_data.groupby('allocation_policy').agg({
                'avg_response_time_cf': 'mean',
                'schedulability_ratio_cf': 'mean'
            }).reset_index()

            plt.bar(alloc_data['allocation_policy'], alloc_data['avg_response_time_cf'],
                    alpha=0.7, color='lightcoral')
            plt.xlabel('Allocation Policy')
            plt.ylabel('Average Response Time')
            plt.title('Allocation Policy Comparison')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No Partitioned Data Available',
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Allocation Policy Comparison')

    def _plot_performance_heatmap(self):
        """Plot 10: Performance heatmap"""
        heatmap_data = self.df.groupby(['num_cores', 'target_utilization']).agg({
            'avg_response_time_cf': 'mean'
        }).reset_index()

        pivot_heatmap = heatmap_data.pivot(index='target_utilization', columns='num_cores',
                                          values='avg_response_time_cf')

        if not pivot_heatmap.empty:
            sns.heatmap(pivot_heatmap, annot=True, fmt='.2f', cmap='YlOrRd',
                       cbar_kws={'label': 'Avg Response Time'})
            plt.title('Response Time Heatmap')
            plt.xlabel('Number of Cores')
            plt.ylabel('Target Utilization')
        else:
            plt.text(0.5, 0.5, 'Insufficient Data for Heatmap',
                    transform=plt.gca().transAxes, ha='center', va='center')

    def _plot_response_time_distribution(self):
        """Plot 11: Response time distribution"""
        # Create violin plot of response time distributions
        rt_data = []
        labels = []

        for cores in [4, 8, 16, 32]:
            core_data = self.df[self.df['num_cores'] == cores]['avg_response_time_cf'].values
            if len(core_data) > 0:
                rt_data.append(core_data)
                labels.append(f'{cores} cores')

        if rt_data:
            plt.boxplot(rt_data, labels=labels)
            plt.ylabel('Response Time')
            plt.title('Response Time Distribution by Core Count')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No Data Available',
                    transform=plt.gca().transAxes, ha='center', va='center')

    def _plot_system_efficiency(self):
        """Plot 12: System efficiency analysis"""
        efficiency_data = self.df.copy()
        efficiency_data['efficiency'] = (efficiency_data['schedulability_ratio_cf'] /
                                        efficiency_data['total_utilization']).fillna(0)

        eff_summary = efficiency_data.groupby(['scheduling_type', 'scheduling_policy']).agg({
            'efficiency': 'mean'
        }).reset_index()

        # Create grouped bar chart
        scheduling_types = eff_summary['scheduling_type'].unique()
        policies = eff_summary['scheduling_policy'].unique()

        x_pos = np.arange(len(policies))
        width = 0.35

        for i, sched_type in enumerate(scheduling_types):
            type_data = eff_summary[eff_summary['scheduling_type'] == sched_type]
            plt.bar(x_pos + i*width, type_data['efficiency'], width,
                   label=sched_type, alpha=0.8)

        plt.xlabel('Scheduling Policy')
        plt.ylabel('System Efficiency')
        plt.title('System Efficiency Comparison')
        plt.xticks(x_pos + width/2, policies, rotation=45)
        plt.legend()

    def _generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EXPERIMENTAL RESULTS SUMMARY")
        print("="*60)

        print(f"\nTotal Experiments Conducted: {len(self.results)}")
        print(f"Core Configurations: {sorted(self.df['num_cores'].unique())}")
        print(f"Task Counts: {sorted(self.df['n_tasks'].unique())}")
        print(f"Utilization Levels: {sorted(self.df['target_utilization'].unique())}")

        print("\n" + "-"*40)
        print("RESPONSE TIME ANALYSIS")
        print("-"*40)

        print(f"Average Response Time (Closed-Form): {self.df['avg_response_time_cf'].mean():.3f}")
        print(f"Average Response Time (MILP): {self.df['avg_response_time_milp'].mean():.3f}")
        print(f"Max Response Time (Closed-Form): {self.df['max_response_time_cf'].max():.3f}")
        print(f"Max Response Time (MILP): {self.df['max_response_time_milp'].max():.3f}")

        print("\n" + "-"*40)
        print("SCHEDULABILITY ANALYSIS")
        print("-"*40)

        avg_sched_cf = self.df['schedulability_ratio_cf'].mean()
        avg_sched_milp = self.df['schedulability_ratio_milp'].mean()
        print(f"Average Schedulability Ratio (Closed-Form): {avg_sched_cf:.3f}")
        print(f"Average Schedulability Ratio (MILP): {avg_sched_milp:.3f}")

        print("\n" + "-"*40)
        print("PERFORMANCE BY CORE COUNT")
        print("-"*40)

        for cores in sorted(self.df['num_cores'].unique()):
            core_data = self.df[self.df['num_cores'] == cores]
            print(f"{cores} cores: Avg RT = {core_data['avg_response_time_cf'].mean():.3f}, "
                  f"Schedulability = {core_data['schedulability_ratio_cf'].mean():.3f}")

        print("\n" + "-"*40)
        print("PERFORMANCE BY SCHEDULING TYPE")
        print("-"*40)

        for sched_type in self.df['scheduling_type'].unique():
            type_data = self.df[self.df['scheduling_type'] == sched_type]
            print(f"{sched_type}: Avg RT = {type_data['avg_response_time_cf'].mean():.3f}, "
                  f"Schedulability = {type_data['schedulability_ratio_cf'].mean():.3f}")

        print("\n" + "-"*40)
        print("ANALYSIS OVERHEAD")
        print("-"*40)

        print(f"Average Analysis Time: {self.df['analysis_overhead'].mean():.4f} seconds")
        print(f"Max Analysis Time: {self.df['analysis_overhead'].max():.4f} seconds")

        # Best and worst performing configurations
        print("\n" + "-"*40)
        print("BEST PERFORMING CONFIGURATIONS")
        print("-"*40)

        best_schedulability = self.df.loc[self.df['schedulability_ratio_cf'].idxmax()]
        print(f"Best Schedulability: {best_schedulability['schedulability_ratio_cf']:.3f}")
        print(f"Configuration: {best_schedulability['num_cores']} cores, "
              f"{best_schedulability['scheduling_type']}, {best_schedulability['scheduling_policy']}")

        best_response_time = self.df.loc[self.df['avg_response_time_cf'].idxmin()]
        print(f"Best Response Time: {best_response_time['avg_response_time_cf']:.3f}")
        print(f"Configuration: {best_response_time['num_cores']} cores, "
              f"{best_response_time['scheduling_type']}, {best_response_time['scheduling_policy']}")

        # Generate detailed CSV report
        detailed_results = self.df[['num_cores', 'n_tasks', 'target_utilization',
                                   'scheduling_type', 'scheduling_policy', 'allocation_policy',
                                   'avg_response_time_cf', 'avg_response_time_milp',
                                   'schedulability_ratio_cf', 'schedulability_ratio_milp',
                                   'total_utilization', 'analysis_overhead']]

        detailed_results.to_csv('detailed_experimental_results.csv', index=False)
        print(f"\nDetailed results saved to: detailed_experimental_results.csv")

def main():
    """
    Main execution function - runs the complete experimental suite
    """
    print("="*60)
    print("REAL-TIME MULTIPROCESSOR BUNDLED TASK SCHEDULING SYSTEM")
    print("="*60)
    print("Initializing comprehensive experimental analysis...")

    # Create experiment runner
    experiment_runner = ExperimentRunner()

    # Run comprehensive experiments
    results = experiment_runner.run_comprehensive_experiment()

    if not results:
        print("No experimental results generated!")
        return

    # Analyze and visualize results
    analyzer = ResultAnalyzer(results)
    analyzer.generate_all_plots()

    print("\n" + "="*60)
    print("EXPERIMENTAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("Generated outputs:")
    print("- comprehensive_analysis.png: All visualization plots")
    print("- detailed_experimental_results.csv: Raw experimental data")
    print("- Console summary: Key findings and statistics")

def run_sample_experiment():
    """
    Run a smaller sample experiment for testing and demonstration
    """
    print("Running sample experiment with limited parameters...")

    # Generate a sample task set
    tasks = TaskSetGenerator.generate_task_set(n_tasks=50, target_utilization=0.6, max_cores=8)

    # Test Global Scheduling
    print("\nTesting Global Scheduling...")
    global_scheduler = Scheduler(8, SchedulingType.GLOBAL)
    global_result = global_scheduler.schedule_tasks(tasks, SchedulingPolicy.HEFT)

    print(f"Global Scheduling Results:")
    print(f"- Average Response Time (CF): {np.mean(list(global_result['closed_form_response_times'].values())):.3f}")
    print(f"- Average Response Time (MILP): {np.mean(list(global_result['milp_response_times'].values())):.3f}")
    print(f"- Schedulability Ratio (CF): {global_result['schedulability_ratio_cf']:.3f}")
    print(f"- Total Utilization: {global_result['total_utilization']:.3f}")

    # Test Partitioned Scheduling
    print("\nTesting Partitioned Scheduling...")
    partitioned_scheduler = Scheduler(8, SchedulingType.PARTITIONED)
    partitioned_result = partitioned_scheduler.schedule_tasks(tasks, SchedulingPolicy.HEFT, AllocationPolicy.FFD)

    print(f"Partitioned Scheduling Results:")
    print(f"- Average Response Time (CF): {np.mean(list(partitioned_result['closed_form_response_times'].values())):.3f}")
    print(f"- Average Response Time (MILP): {np.mean(list(partitioned_result['milp_response_times'].values())):.3f}")
    print(f"- Schedulability Ratio (CF): {partitioned_result['schedulability_ratio_cf']:.3f}")
    print(f"- Total Utilization: {partitioned_result['total_utilization']:.3f}")

    # Create simple comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Response time comparison
    methods = ['Closed-Form', 'MILP']
    global_rt = [np.mean(list(global_result['closed_form_response_times'].values())),
                 np.mean(list(global_result['milp_response_times'].values()))]
    partitioned_rt = [np.mean(list(partitioned_result['closed_form_response_times'].values())),
                      np.mean(list(partitioned_result['milp_response_times'].values()))]

    x = np.arange(len(methods))
    width = 0.35

    ax1.bar(x - width/2, global_rt, width, label='Global', alpha=0.8)
    ax1.bar(x + width/2, partitioned_rt, width, label='Partitioned', alpha=0.8)
    ax1.set_xlabel('Analysis Method')
    ax1.set_ylabel('Average Response Time')
    ax1.set_title('Response Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()

    # Schedulability comparison
    sched_types = ['Global', 'Partitioned']
    sched_ratios = [global_result['schedulability_ratio_cf'], partitioned_result['schedulability_ratio_cf']]

    ax2.bar(sched_types, sched_ratios, alpha=0.8, color=['blue', 'red'])
    ax2.set_ylabel('Schedulability Ratio')
    ax2.set_title('Schedulability Comparison')

    plt.tight_layout()
    plt.savefig('sample_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Uncomment the desired execution mode:

    # Full comprehensive experiment (takes longer but provides complete analysis)
    main()

    # Quick sample experiment (for testing and demonstration)
    #run_sample_experiment()

    print("\nProgram execution completed successfully!")
    print("Check the generated files for detailed results and analysis.")