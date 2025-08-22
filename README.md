# Response-Time Analysis of Gang Task Scheduling in Real-Time Multiprocessor Systems

## 📖 Overview
Real-time multiprocessor systems are at the core of **cyber-physical systems**, **industrial control**, and **embedded processing**. These systems require efficient task scheduling to minimize unpredictable delays and maximize reliability.  

This project investigates **response-time analysis methods** for *gang (bundled) task scheduling* in heterogeneous multiprocessor systems with 4, 8, 16, and 32 cores. We compare analytical and optimization-based approaches under different scheduling policies and allocation strategies.

The ultimate goal is to provide **effective models and methods for analyzing and optimizing bundled task scheduling**, improving utilization, predictability, and energy efficiency in modern real-time systems.

---

## 🎯 Objectives
- Develop and compare two **response-time analysis (RTA)** approaches:
  1. **Closed-Form Analysis** (analytical method)  
  2. **Optimization-Based Analysis** using **Mixed-Integer Linear Programming (MILP)**
- Evaluate **Partitioned Scheduling** and **Global Scheduling** under the **Bundled Tasks model**.
- Test multiple **task-to-core allocation strategies**:
  - Decreasing Worst-Fit
  - Best-Fit Decreasing
  - First-Fit Decreasing
  - Round-Robin
- Investigate the scalability of schedulability as the number of tasks increases (10 → 500).

---

## ⚙️ Problem Setting
### 🔹 Task Parameters
Each task is described by:
- **Worst-Case Execution Time (WCET)**
- **Period**
- **Number of Bundles (sequential segments)**
- **Core requirements** (dynamic, varying with segments)
- **Scheduling policies**: LDF, HEFT, FIFO
- **Preemptive / Non-Preemptive models**

### 🔹 System Parameters
- Multiprocessors with **4, 8, 16, 32 cores**
- Heterogeneous processing speeds
- Varying utilization levels: `[0.25, 0.5, 0.75]`
- Number of tasks: `n ∈ {10, 50, 100, 200, 400, 600}`

---

## 📊 Expected Outputs
1. **Response-time plots** under different RTA methods.  
2. **Schedulability comparison** across 4, 8, 16, and 32 cores.  
3. **Impact of utilization** on response time and energy consumption.  
4. **Analysis of task migration** and scheduling-induced delays.  
5. **Comparative overhead plots** for Closed-Form vs. MILP-based RTA.  
6. Final **comprehensive report**, including:
   - Line-by-line explanation of the implemented code.  
   - Charts and figures with full analysis.  

---

## 🛠️ Implementation Plan
1. Implement **task generation** with parameters and bundles.  
2. Implement **Closed-Form RTA**.  
3. Implement **MILP-based RTA**.  
4. Implement allocation strategies (Worst-Fit, Best-Fit, First-Fit, Round-Robin).  
5. Run simulations across cores = {4, 8, 16, 32}.  
6. Collect and visualize metrics:
   - Deadline miss ratios
   - Average/maximum response times
   - Utilization and energy efficiency
7. Prepare final **report with results**.

---

## 📈 Evaluation Metrics
- **Schedulability rate** (% tasks meeting deadlines)  
- **Average and maximum response time**  
- **Processor utilization efficiency**  
- **Energy consumption trends**  
- **Computational overhead** of analysis methods  

---

## 👩‍💻 Maintainers
- **Mahdi Saieedi** (@Maximus42-byte)  
- **Mahnoosh Ramtin** (@mahnooshr) 
---

## 📌 Notes
- All simulation code must be **fully documented** (line-by-line).  
- The **final report is mandatory**, with detailed result analysis and visualizations.  
