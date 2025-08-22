import numpy as np
import torch
import torch.cuda as cuda
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import time
from collections import defaultdict
import seaborn as sns
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set default tensor type to CUDA if available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

@dataclass
class Bundle:
    """Represents a bundle (segment) within a task"""
    execution_time: float
    required_cores: int
    
@dataclass
class Task:
    """Represents a real-time task with multiple bundles"""
    id: int
    period: float
    deadline: float
    bundles: List[Bundle]
    priority: float = 0.0
    
    def __post_init__(self):
        self.wcet = sum(bundle.execution_time for bundle in self.bundles)
        self.max_parallelism = max(bundle.required_cores for bundle in self.bundles)
        self.utilization = self.wcet / self.period

class GPUTaskData:
    """GPU-friendly task data structure using PyTorch"""
    def __init__(self, tasks: List[Task]):
        self.num_tasks = len(tasks)
        
        # Convert to GPU tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.periods = torch.tensor([t.period for t in tasks], dtype=torch.float32, device=device)
        self.wcets = torch.tensor([t.wcet for t in tasks], dtype=torch.float32, device=device)
        self.deadlines = torch.tensor([t.deadline for t in tasks], dtype=torch.float32, device=device)
        self.priorities = torch.tensor([t.priority for t in tasks], dtype=torch.float32, device=device)
        self.max_parallelisms = torch.tensor([t.max_parallelism for t in tasks], dtype=torch.int32, device=device)
        self.utilizations = torch.tensor([t.utilization for t in tasks], dtype=torch.float32, device=device)

class Processor:
    """Represents a processing core"""
    def __init__(self, id: int, speed: float = 1.0):
        self.id = id
        self.speed = speed
        self.allocated_tasks = []
        self.utilization = 0.0

class MultiprocessorSystem:
    """Represents the multiprocessor system"""
    def __init__(self, num_cores: int, heterogeneous: bool = True):
        self.num_cores = num_cores
        if heterogeneous:
            self.processors = [Processor(i, 0.8 + 0.4 * random.random()) 
                             for i in range(num_cores)]
        else:
            self.processors = [Processor(i) for i in range(num_cores)]
            
    def reset_allocations(self):
        for proc in self.processors:
            proc.allocated_tasks = []
            proc.utilization = 0.0

class TaskGenerator:
    """Generates synthetic real-time tasks"""
    @staticmethod
    def generate_tasks(num_tasks: int, target_utilization: float, 
                      num_cores: int) -> List[Task]:
        tasks = []
        
        for i in range(num_tasks):
            period = random.uniform(100, 1000)
            num_bundles = random.randint(1, 3)
            
            bundles = []
            remaining_util = target_utilization / num_tasks
            
            for j in range(num_bundles):
                exec_time = random.uniform(0.1, 0.3) * period * remaining_util
                req_cores = random.randint(1, min(4, num_cores))
                bundles.append(Bundle(exec_time, req_cores))
            
            task = Task(id=i, period=period, deadline=period, bundles=bundles)
            tasks.append(task)
            
        return tasks

class TorchResponseTimeAnalysis:
    """GPU-accelerated response time analysis using PyTorch"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA not available, using CPU")
    
    def analyze_batch(self, tasks: List[Task], system) -> np.ndarray:
        """Analyze all tasks in batch on GPU"""
        gpu_data = GPUTaskData(tasks)
        num_tasks = len(tasks)
        
        # Initialize response times
        response_times = torch.zeros(num_tasks, device=self.device)
        
        # Process each task
        for i in range(num_tasks):
            response_times[i] = self._analyze_single_gpu(
                i, gpu_data, system.num_cores
            )
        
        return response_times.cpu().numpy()
    
    def _analyze_single_gpu(self, task_idx: int, gpu_data: GPUTaskData, 
                           num_cores: int) -> float:
        """Analyze single task using GPU tensors"""
        task_wcet = gpu_data.wcets[task_idx]
        task_priority = gpu_data.priorities[task_idx]
        task_deadline = gpu_data.deadlines[task_idx]
        
        # Find higher priority tasks
        hp_mask = gpu_data.priorities > task_priority
        
        if not hp_mask.any():
            return task_wcet.item()
        
        # Binary search for response time
        R_min = task_wcet
        R_max = task_deadline * 2.0
        
        for _ in range(20):  # Fixed iterations
            R = (R_min + R_max) / 2.0
            
            # Calculate interference using vectorized operations
            num_instances = torch.ceil(R / gpu_data.periods)
            interference_per_task = num_instances * gpu_data.wcets
            
            # Parallel execution factor
            parallel_factors = torch.minimum(
                gpu_data.max_parallelisms.float(), 
                torch.tensor(float(num_cores), device=self.device)
            ) / num_cores
            
            # Total interference from higher priority tasks
            total_interference = (interference_per_task * parallel_factors * hp_mask).sum()
            
            R_new = task_wcet + total_interference
            
            if R_new <= R:
                R_max = R
            else:
                R_min = R
        
        return R_max.item()
    
    def check_schedulability_batch(self, response_times: torch.Tensor, 
                                  deadlines: torch.Tensor) -> int:
        """Check schedulability for all tasks on GPU"""
        schedulable = response_times <= deadlines
        return schedulable.sum().item()

class TorchAllocationStrategy:
    """GPU-accelerated allocation using PyTorch"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def allocate_gpu(self, tasks: List[Task], system):
        """Allocate tasks using GPU acceleration"""
        num_tasks = len(tasks)
        num_processors = system.num_cores
        
        # Convert to tensors
        task_utils = torch.tensor([t.utilization for t in tasks], 
                                 device=self.device, dtype=torch.float32)
        
        # Sort tasks by utilization (descending)
        sorted_utils, sorted_indices = torch.sort(task_utils, descending=True)
        
        # CPU allocation (simpler than full GPU implementation)
        system.reset_allocations()
        processor_utils = [0.0] * num_processors
        
        for i in range(num_tasks):
            task_idx = sorted_indices[i].item()
            task_util = sorted_utils[i].item()
            
            # Find best processor
            best_proc = -1
            min_util = float('inf')
            
            for p in range(num_processors):
                if processor_utils[p] + task_util <= 1.0:
                    if processor_utils[p] < min_util:
                        min_util = processor_utils[p]
                        best_proc = p
            
            if best_proc >= 0:
                system.processors[best_proc].allocated_tasks.append(tasks[task_idx])
                system.processors[best_proc].utilization += task_util
                processor_utils[best_proc] += task_util

class SchedulingPolicy:
    """Base class for scheduling policies"""
    def assign_priorities(self, tasks: List[Task]):
        raise NotImplementedError

class LDF(SchedulingPolicy):
    """Least Deadline First"""
    def assign_priorities(self, tasks: List[Task]):
        sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
        for i, task in enumerate(sorted_tasks):
            task.priority = len(tasks) - i

class HEFT(SchedulingPolicy):
    """Heterogeneous Earliest Finish Time"""
    def assign_priorities(self, tasks: List[Task]):
        for task in tasks:
            task.priority = task.utilization

class FIFO(SchedulingPolicy):
    """First In First Out"""
    def assign_priorities(self, tasks: List[Task]):
        for i, task in enumerate(tasks):
            task.priority = len(tasks) - i

class TorchExperimentRunner:
    """GPU-accelerated experiment runner using PyTorch"""
    def __init__(self):
        self.results = defaultdict(list)
        self.torch_analysis = TorchResponseTimeAnalysis()
        self.torch_allocation = TorchAllocationStrategy()
        
    def run_experiment(self, num_tasks: int, num_cores: int, 
                      utilization: float):
        """Run a single experiment configuration"""
        # Generate tasks
        tasks = TaskGenerator.generate_tasks(num_tasks, utilization, num_cores)
        
        # Create system
        system = MultiprocessorSystem(num_cores)
        
        scheduling_policies = {
            'LDF': LDF(),
            'HEFT': HEFT(),
            'FIFO': FIFO()
        }
        
        allocation_methods = {
            'GPU-Optimized': self.torch_allocation
        }
        
        results = []
        
        for alloc_name, alloc_method in allocation_methods.items():
            for sched_name, sched_policy in scheduling_policies.items():
                # Allocate tasks
                start_time = time.time()
                alloc_method.allocate_gpu(tasks, system)
                
                # Assign priorities
                sched_policy.assign_priorities(tasks)
                
                # Analyze all tasks in batch on GPU
                response_times = self.torch_analysis.analyze_batch(tasks, system)
                
                # Check schedulability
                schedulable_count = np.sum(response_times <= np.array([t.deadline for t in tasks]))
                
                computation_time = time.time() - start_time
                
                result = {
                    'num_tasks': num_tasks,
                    'num_cores': num_cores,
                    'utilization': utilization,
                    'allocation': alloc_name,
                    'scheduling': sched_name,
                    'analysis': 'PyTorch-GPU' if self.torch_analysis.use_gpu else 'PyTorch-CPU',
                    'avg_response_time': np.mean(response_times),
                    'max_response_time': np.max(response_times),
                    'schedulability_ratio': schedulable_count / num_tasks,
                    'computation_time': computation_time,
                    'system_utilization': np.mean([p.utilization for p in system.processors])
                }
                
                results.append(result)
        
        return results
    
    def run_all_experiments(self, task_counts, core_counts, utilizations):
        """Run all experiment configurations"""
        total_configs = len(task_counts) * len(core_counts) * len(utilizations)
        config_count = 0
        
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Running {total_configs} configurations on {device_name}...")
        
        for num_tasks in task_counts:
            for num_cores in core_counts:
                for utilization in utilizations:
                    config_count += 1
                    print(f"Configuration {config_count}/{total_configs}: "
                          f"Tasks={num_tasks}, Cores={num_cores}, Util={utilization}")
                    
                    results = self.run_experiment(num_tasks, num_cores, utilization)
                    self.results['experiments'].extend(results)
                    
                    # Clear GPU cache periodically
                    if torch.cuda.is_available() and config_count % 10 == 0:
                        torch.cuda.empty_cache()
        
        print("\nAll experiments completed!")
    
    def plot_results(self):
        """Generate visualization plots"""
        experiments = self.results['experiments']
        
        # Convert to pandas DataFrame
        import pandas as pd
        df = pd.DataFrame(experiments)
        
        # 1. Response time by task count
        plt.figure(figsize=(12, 8))
        avg_rt = df.groupby('num_tasks')['avg_response_time'].mean()
        plt.plot(avg_rt.index, avg_rt.values, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Tasks', fontsize=12)
        plt.ylabel('Average Response Time', fontsize=12)
        plt.title('PyTorch GPU-Accelerated Response Time Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pytorch_response_time.png', dpi=300)
        plt.close()
        
        # 2. Schedulability vs cores
        plt.figure(figsize=(12, 8))
        for num_cores in sorted(df['num_cores'].unique()):
            data = df[df['num_cores'] == num_cores]
            avg_sched = data.groupby('num_tasks')['schedulability_ratio'].mean()
            plt.plot(avg_sched.index, avg_sched.values, 
                    marker='o', label=f'{num_cores} cores', linewidth=2)
        
        plt.xlabel('Number of Tasks', fontsize=12)
        plt.ylabel('Schedulability Ratio', fontsize=12)
        plt.title('PyTorch GPU-Accelerated Schedulability Analysis', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pytorch_schedulability.png', dpi=300)
        plt.close()
        
        # 3. Computation time
        plt.figure(figsize=(12, 8))
        avg_time = df.groupby('num_tasks')['computation_time'].mean()
        plt.plot(avg_time.index, avg_time.values, 'g-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Tasks', fontsize=12)
        plt.ylabel('Computation Time (seconds)', fontsize=12)
        plt.title('PyTorch GPU Computation Time Scaling', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('pytorch_computation_time.png', dpi=300)
        plt.close()
        
        # 4. Utilization impact
        plt.figure(figsize=(12, 8))
        for util in sorted(df['utilization'].unique()):
            data = df[df['utilization'] == util]
            avg_rt = data.groupby('num_tasks')['avg_response_time'].mean()
            plt.plot(avg_rt.index, avg_rt.values, 
                    marker='o', label=f'Utilization {util}', linewidth=2)
        
        plt.xlabel('Number of Tasks', fontsize=12)
        plt.ylabel('Average Response Time', fontsize=12)
        plt.title('Impact of Utilization on Response Time (PyTorch GPU)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pytorch_utilization_impact.png', dpi=300)
        plt.close()
        
        # 5. Heatmap
        pivot_data = df.pivot_table(
            values='schedulability_ratio',
            index='utilization',
            columns='num_cores',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Schedulability Ratio'})
        plt.title('Schedulability Heatmap (PyTorch GPU)', fontsize=14)
        plt.xlabel('Number of Cores', fontsize=12)
        plt.ylabel('Utilization', fontsize=12)
        plt.tight_layout()
        plt.savefig('pytorch_schedulability_heatmap.png', dpi=300)
        plt.close()
        
        print("All plots generated!")

def print_gpu_info():
    """Print GPU information using PyTorch"""
    print("\n" + "="*60)
    print("GPU INFORMATION (PyTorch)")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print(f"  CUDA Cores/MP: {props.multi_processor_count * 128}")  # Estimate
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {allocated:.2f} GB")
            print(f"  Memory Reserved: {reserved:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")
    
    print("="*60 + "\n")

def generate_report(results):
    """Generate summary report"""
    experiments = results['experiments']
    
    with open('pytorch_gang_scheduling_report.txt', 'w') as f:
        f.write("PyTorch GPU-Accelerated Gang Task Scheduling Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        # GPU info
        if torch.cuda.is_available():
            f.write(f"GPU Used: {torch.cuda.get_device_name(0)}\n")
            props = torch.cuda.get_device_properties(0)
            f.write(f"Compute Capability: {props.major}.{props.minor}\n")
            f.write(f"Total GPU Memory: {props.total_memory / 1024**3:.2f} GB\n\n")
        else:
            f.write("Running on CPU (CUDA not available)\n\n")
        
        # Performance summary
        f.write("Performance Summary\n")
        f.write("-" * 30 + "\n")
        
        total_time = sum(exp['computation_time'] for exp in experiments)
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Total computation time: {total_time:.2f} seconds\n")
        f.write(f"Average time per experiment: {total_time/len(experiments):.4f} seconds\n\n")
        
        # Schedulability by task count
        f.write("Schedulability by Task Count\n")
        f.write("-" * 30 + "\n")
        
        task_performance = defaultdict(list)
        for exp in experiments:
            task_performance[exp['num_tasks']].append(exp['schedulability_ratio'])
        
        for num_tasks in sorted(task_performance.keys()):
            avg_sched = np.mean(task_performance[num_tasks])
            f.write(f"{num_tasks} tasks: {avg_sched:.3f} avg schedulability\n")
        
        # Best configurations
        f.write("\nTop 5 Configurations by Schedulability\n")
        f.write("-" * 30 + "\n")
        
        config_performance = defaultdict(list)
        for exp in experiments:
            key = f"{exp['allocation']}-{exp['scheduling']}"
            config_performance[key].append(exp['schedulability_ratio'])
        
        sorted_configs = sorted(config_performance.items(), 
                               key=lambda x: np.mean(x[1]), reverse=True)
        
        for i, (config, values) in enumerate(sorted_configs[:5], 1):
            f.write(f"{i}. {config}: {np.mean(values):.3f} avg schedulability\n")
        
        print("Report generated: pytorch_gang_scheduling_report.txt")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("PYTORCH GPU-ACCELERATED GANG TASK SCHEDULING ANALYSIS")
    print("="*60)
    
    # Print GPU information
    print_gpu_info()
    
    start_time = time.time()
    
    runner = TorchExperimentRunner()
    
    # Test configurations
    task_counts = [10, 50, 100, 200, 400, 600]
    core_counts = [4, 8, 16, 32]
    utilizations = [0.25, 0.5, 0.75]
    
    # Run experiments
    runner.run_all_experiments(task_counts, core_counts, utilizations)
    
    # Generate plots
    print("\nGenerating plots...")
    runner.plot_results()
    
    # Generate report
    print("\nGenerating report...")
    generate_report(runner.results)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Experiments completed: {len(runner.results['experiments'])}")
    print(f"Average time per experiment: {total_time/len(runner.results['experiments']):.4f} seconds")
    
    # Show GPU memory usage
    if torch.cuda.is_available():
        print(f"\nPeak GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()