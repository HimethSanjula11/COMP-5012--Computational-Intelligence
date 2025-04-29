"""
Multi-Objective Genetic Algorithm (MOGA) for Personnel Shift Scheduling
This implementation demonstrates a MOGA approach to solving the shift minimization 
personnel task scheduling problem from the Operations Research library.

The algorithm optimizes for three objectives:
1. Minimizing the number of shifts (workers used)
2. Maximizing job coverage
3. Minimizing overlapping jobs per worker

The implementation includes:
- Problem representation
- Objective functions
- Genetic operators (selection, crossover, mutation)
- Archive for maintaining the Pareto front
- Visualization tools for analyzing results
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import pandas as pd
import re
import time
from datetime import datetime

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Classes for representing the problem
class Job:
    def __init__(self, id, start_time, end_time):
        """
        Initialize a job with its timing constraints.
        
        Args:
            id: Unique identifier for the job
            start_time: Earliest start time for the job
            end_time: Latest end time for the job
        """
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
    
    def overlaps(self, other_job):
        """
        Check if this job overlaps in time with another job.
        
        Args:
            other_job: Another job to check overlap with
            
        Returns:
            bool: True if the jobs overlap, False otherwise
        """
        return (self.start_time < other_job.end_time and
                self.end_time > other_job.start_time)

class Worker:
    def __init__(self, id, qualifications):
        """
        Initialize a worker with their qualifications.
        
        Args:
            id: Unique identifier for the worker
            qualifications: List of job IDs the worker is qualified to perform
        """
        self.id = id
        self.qualifications = qualifications
    
    def can_perform(self, job_id):
        """
        Check if the worker can perform a specific job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            bool: True if worker is qualified for the job, False otherwise
        """
        return job_id in self.qualifications

# Solution representation
class Individual:
    def __init__(self, num_workers, num_jobs):
        """
        Initialize a solution individual.
        
        Args:
            num_workers: Number of available workers
            num_jobs: Number of jobs to be scheduled
        """
        # Assignment matrix: rows=workers, columns=jobs
        self.assignment_matrix = np.zeros((num_workers, num_jobs), dtype=int)
        self.fitness_values = None
        self.creation_time = time.time()  # Track when this solution was created
    
    def initialize_random(self, workers, jobs):
        """
        Create a random initial solution by assigning each job to a random qualified worker.
        
        Args:
            workers: List of Worker objects
            jobs: List of Job objects
        """
        for j_idx in range(len(jobs)):
            # Find qualified workers for this job
            qualified_workers = [w.id for w in workers if w.can_perform(j_idx)]
            if qualified_workers:
                # Assign job to a random qualified worker
                worker_id = random.choice(qualified_workers)
                self.assignment_matrix[worker_id, j_idx] = 1
    
    def calculate_fitness(self, workers, jobs):
        """
        Calculate the multiple objectives for this solution.
        
        Args:
            workers: List of Worker objects
            jobs: List of Job objects
            
        Returns:
            list: The fitness values [shifts, -coverage, overlap]
        """
        # Objective 1: Minimize number of shifts (workers that are assigned at least one job)
        shifts_used = np.sum(np.any(self.assignment_matrix, axis=1))
        
        # Objective 2: Maximize job coverage
        jobs_covered = np.sum(np.any(self.assignment_matrix, axis=0))
        
        # Objective 3: Minimize overlapping jobs per worker
        overlap_penalty = 0
        for w_idx in range(len(workers)):
            assigned_jobs = [j for j in range(len(jobs)) if self.assignment_matrix[w_idx, j] == 1]
            for i, j1_idx in enumerate(assigned_jobs):
                for j2_idx in assigned_jobs[i+1:]:
                    if jobs[j1_idx].overlaps(jobs[j2_idx]):
                        overlap_penalty += 1
        
        # Store and return fitness values (minimize shifts and overlap, maximize coverage)
        # Note: We negate coverage to convert maximization to minimization
        self.fitness_values = [shifts_used, -jobs_covered, overlap_penalty]
        return self.fitness_values
    
    def repair(self, workers, jobs):
        """
        Repair invalid solutions by ensuring workers are qualified and resolving overlaps.
        
        This function implements a three-step repair strategy:
        1. Ensure jobs are assigned only to qualified workers
        2. Resolve overlapping jobs for each worker using a greedy algorithm
        3. Re-assign any jobs that became unassigned
        
        Args:
            workers: List of Worker objects
            jobs: List of Job objects
        """
        # Step 1: Ensure all jobs are assigned to qualified workers
        for w_idx in range(len(workers)):
            for j_idx in range(len(jobs)):
                if (self.assignment_matrix[w_idx, j_idx] == 1 and 
                    not workers[w_idx].can_perform(j_idx)):
                    self.assignment_matrix[w_idx, j_idx] = 0
        
        # Step 2: Resolve overlapping jobs for each worker
        for w_idx in range(len(workers)):
            assigned_jobs = [(j_idx, jobs[j_idx]) 
                            for j_idx in range(len(jobs)) 
                            if self.assignment_matrix[w_idx, j_idx] == 1]
            
            # Sort jobs by end time (earliest finish time first)
            assigned_jobs.sort(key=lambda x: x[1].end_time)
            
            # Greedy algorithm to select non-overlapping jobs
            selected = []
            for j_idx, job in assigned_jobs:
                if not any(job.overlaps(selected_job[1]) for selected_job in selected):
                    selected.append((j_idx, job))
                else:
                    self.assignment_matrix[w_idx, j_idx] = 0
        
        # Step 3: Ensure all jobs are assigned
        for j_idx in range(len(jobs)):
            if np.sum(self.assignment_matrix[:, j_idx]) == 0:
                # Find qualified workers for this job
                qualified_workers = [w.id for w in workers if w.can_perform(j_idx)]
                if qualified_workers:
                    # Assign to a random qualified worker
                    worker_id = random.choice(qualified_workers)
                    self.assignment_matrix[worker_id, j_idx] = 1
    
    def get_assignments(self):
        """
        Get the worker-job assignments in this solution.
        
        Returns:
            list: Tuples of (worker_id, job_id) for each assignment
        """
        assignments = []
        for w_idx in range(self.assignment_matrix.shape[0]):
            for j_idx in range(self.assignment_matrix.shape[1]):
                if self.assignment_matrix[w_idx, j_idx] == 1:
                    assignments.append((w_idx, j_idx))
        return assignments
    
    def get_worker_load(self, jobs):
        """
        Calculate the workload for each worker.
        
        Args:
            jobs: List of Job objects
            
        Returns:
            dict: Mapping from worker_id to total working time
        """
        worker_load = {}
        for w_idx in range(self.assignment_matrix.shape[0]):
            load = 0
            for j_idx in range(self.assignment_matrix.shape[1]):
                if self.assignment_matrix[w_idx, j_idx] == 1:
                    load += jobs[j_idx].duration
            if load > 0:  # Only include workers with assignments
                worker_load[w_idx] = load
        return worker_load

# Dominance function for comparing solutions
def dominates(fitness1, fitness2):
    """
    Check if fitness1 dominates fitness2 (is better in at least one objective and not worse in any).
    
    Args:
        fitness1: First fitness vector
        fitness2: Second fitness vector
        
    Returns:
        bool: True if fitness1 dominates fitness2, False otherwise
    """
    # Check if either fitness is None
    if fitness1 is None or fitness2 is None:
        return False
        
    better_in_at_least_one = False
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:  # If any objective is worse (assuming minimization)
            return False
        if f1 < f2:  # If any objective is better
            better_in_at_least_one = True
    return better_in_at_least_one

# Archive to store non-dominated solutions
class Archive:
    def __init__(self, max_size=50):
        """
        Initialize an archive to store the Pareto front of non-dominated solutions.
        
        Args:
            max_size: Maximum number of solutions to keep in the archive
        """
        self.solutions = []
        self.max_size = max_size
        self.dominance_history = []  # Track how solutions enter and leave the archive
    
    def add(self, individual):
        """
        Add an individual to the archive if it's non-dominated.
        
        Args:
            individual: Solution to potentially add to the archive
            
        Returns:
            bool: True if added to archive, False otherwise
        """
        # Check if individual is dominated by any in archive
        for sol in self.solutions:
            if dominates(sol.fitness_values, individual.fitness_values):
                return False
        
        # Record solutions that will be removed
        removed = [i for i, sol in enumerate(self.solutions) 
                  if dominates(individual.fitness_values, sol.fitness_values)]
        
        # Remove solutions that are dominated by the new individual
        self.solutions = [sol for sol in self.solutions 
                          if not dominates(individual.fitness_values, sol.fitness_values)]
        
        # Add new individual
        self.solutions.append(copy.deepcopy(individual))
        
        # Record this archive change
        self.dominance_history.append({
            'added': individual.fitness_values,
            'removed_count': len(removed),
            'time': time.time()
        })
        
        # If archive exceeds max size, perform pruning
        if len(self.solutions) > self.max_size:
            self._prune_archive()
        
        return True
    
    def _prune_archive(self):
        """
        Reduce archive size by removing crowded solutions with smallest crowding distance.
        """
        # Calculate crowding distance for each solution
        distances = self._calculate_crowding_distance()
        
        # Sort by crowding distance (ascending)
        sorted_indices = np.argsort(distances)
        
        # Remove solutions with smallest crowding distance
        to_remove = len(self.solutions) - self.max_size
        self.solutions = [self.solutions[i] for i in sorted_indices[to_remove:]]
    
    def _calculate_crowding_distance(self):
        """
        Calculate crowding distance for each solution.
        
        Returns:
            numpy.ndarray: Array of crowding distances
        """
        num_solutions = len(self.solutions)
        # Handle special cases
        if num_solutions <= 2:
            return np.array([float('inf')] * num_solutions)
            
        # Filter out solutions with None fitness values
        valid_solutions = [sol for sol in self.solutions if sol.fitness_values is not None]
        if not valid_solutions:
            return np.zeros(num_solutions)
            
        num_objectives = len(valid_solutions[0].fitness_values)
        distances = np.zeros(num_solutions)
        
        # Map from original indices to valid solution indices
        valid_indices = {i: j for j, i in enumerate([self.solutions.index(sol) for sol in valid_solutions])}
        
        for obj in range(num_objectives):
            try:
                # Extract fitness values for this objective
                fitness = [sol.fitness_values[obj] for sol in valid_solutions]
                
                # Skip if all values are the same
                if len(set(fitness)) <= 1:
                    continue
                
                # Sort indices by this objective
                sorted_indices = np.argsort(fitness)
                
                # Set distance for boundary solutions to infinity
                original_min_idx = self.solutions.index(valid_solutions[sorted_indices[0]])
                original_max_idx = self.solutions.index(valid_solutions[sorted_indices[-1]])
                distances[original_min_idx] = float('inf')
                distances[original_max_idx] = float('inf')
                
                # Calculate distance for other solutions
                fitness_range = max(fitness) - min(fitness)
                if fitness_range <= 0:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    original_idx = self.solutions.index(valid_solutions[sorted_indices[i]])
                    value = (fitness[sorted_indices[i+1]] - fitness[sorted_indices[i-1]]) / fitness_range
                    distances[original_idx] += value
            except Exception as e:
                print(f"Error in crowding distance calculation: {e}")
                continue
        
        return distances
    
    def get_statistics(self):
        """
        Calculate statistics about the current archive.
        
        Returns:
            dict: Various statistics about the archive solutions
        """
        if not self.solutions:
            return {}
            
        stats = {
            'size': len(self.solutions),
            'min_shifts': min(sol.fitness_values[0] for sol in self.solutions if sol.fitness_values),
            'max_shifts': max(sol.fitness_values[0] for sol in self.solutions if sol.fitness_values),
            'min_overlap': min(sol.fitness_values[2] for sol in self.solutions if sol.fitness_values),
            'max_overlap': max(sol.fitness_values[2] for sol in self.solutions if sol.fitness_values)
        }
        
        return stats

# Genetic operators
def tournament_selection(population, tournament_size=3):
    """
    Select an individual using tournament selection.
    
    Args:
        population: List of individuals to select from
        tournament_size: Number of candidates to include in each tournament
        
    Returns:
        Individual: The selected individual
    """
    # Avoid error if population is too small
    if len(population) < tournament_size:
        return random.choice(population)
        
    candidates = random.sample(population, tournament_size)
    best_candidate = candidates[0]
    for candidate in candidates[1:]:
        if candidate.fitness_values and best_candidate.fitness_values and dominates(candidate.fitness_values, best_candidate.fitness_values):
            best_candidate = candidate
    return best_candidate

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Perform crossover between two parents to create a child solution.
    
    This implementation uses uniform crossover, where each gene has an equal
    probability of coming from either parent.
    
    Args:
        parent1: First parent solution
        parent2: Second parent solution
        crossover_rate: Probability of performing crossover
        
    Returns:
        Individual: The child solution
    """
    if random.random() > crossover_rate:
        return copy.deepcopy(parent1)
    
    # Create child with same dimensions
    child = Individual(parent1.assignment_matrix.shape[0], parent1.assignment_matrix.shape[1])
    
    # Uniform crossover
    for w in range(parent1.assignment_matrix.shape[0]):
        for j in range(parent1.assignment_matrix.shape[1]):
            if random.random() < 0.5:
                child.assignment_matrix[w, j] = parent1.assignment_matrix[w, j]
            else:
                child.assignment_matrix[w, j] = parent2.assignment_matrix[w, j]
    
    return child

def mutation(individual, workers, jobs, mutation_rate=0.1):
    """
    Perform mutation on an individual by reassigning random jobs.
    
    Args:
        individual: The solution to mutate
        workers: List of Worker objects
        jobs: List of Job objects
        mutation_rate: Probability of mutating each job
        
    Returns:
        Individual: The mutated solution
    """
    for j in range(len(jobs)):
        if random.random() < mutation_rate:
            # Find qualified workers for this job
            qualified_workers = [w.id for w in workers if w.can_perform(j)]
            if qualified_workers:
                # Clear current assignment
                for w in range(len(workers)):
                    individual.assignment_matrix[w, j] = 0
                
                # Make new assignment
                new_worker = random.choice(qualified_workers)
                individual.assignment_matrix[new_worker, j] = 1
    
    return individual 

# Main MOGA algorithm
def run_moga(workers, jobs, population_size=100, generations=100, archive_size=50, 
             crossover_rate=0.8, mutation_rate=0.1, tournament_size=3, seed=None):
    """
    Run the Multi-Objective Genetic Algorithm.
    
    Args:
        workers: List of Worker objects
        jobs: List of Job objects
        population_size: Number of solutions in each generation
        generations: Number of generations to evolve
        archive_size: Maximum size of the Pareto front archive
        crossover_rate: Probability of performing crossover
        mutation_rate: Probability of mutating each job
        tournament_size: Number of candidates in tournament selection
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (archive, history, stats)
            - archive: The final Pareto front archive
            - history: Dictionary of progress metrics for each generation
            - stats: Dictionary of algorithm statistics
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Performance tracking
    start_time = time.time()
    repair_time = 0
    fitness_time = 0
    selection_time = 0
    crossover_time = 0
    mutation_time = 0
    
    # Initialize population
    population = []
    archive = Archive(max_size=archive_size)
    history = []
    
    # Create initial random population
    print("Generating initial population...")
    for i in range(population_size):
        ind = Individual(len(workers), len(jobs))
        ind.initialize_random(workers, jobs)
        
        # Time repair operations
        t0 = time.time()
        ind.repair(workers, jobs)
        repair_time += time.time() - t0
        
        # Time fitness calculations
        t0 = time.time()
        ind.calculate_fitness(workers, jobs)
        fitness_time += time.time() - t0
        
        population.append(ind)
        archive.add(ind)
        
        # Progress indicator
        if (i+1) % 10 == 0:
            print(f"  Created {i+1}/{population_size} initial solutions")
    
    print("Initial population created.")
    
    # Evolution
    for gen in range(generations):
        gen_start = time.time()
        new_population = []
        
        # Elitism - add some solutions directly from archive
        elite_count = min(10, len(archive.solutions))
        if elite_count > 0:
            valid_elites = [sol for sol in archive.solutions if sol.fitness_values is not None]
            if valid_elites:
                elites = random.sample(valid_elites, min(elite_count, len(valid_elites)))
                new_population.extend([copy.deepcopy(elite) for elite in elites])
        
        # Generate rest of population
        while len(new_population) < population_size:
            # Selection
            t0 = time.time()
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            selection_time += time.time() - t0
            
            # Crossover
            t0 = time.time()
            child = crossover(parent1, parent2, crossover_rate)
            crossover_time += time.time() - t0
            
            # Mutation
            t0 = time.time()
            child = mutation(child, workers, jobs, mutation_rate)
            mutation_time += time.time() - t0
            
            # Repair
            t0 = time.time()
            child.repair(workers, jobs)
            repair_time += time.time() - t0
            
            # Calculate fitness
            t0 = time.time()
            child.calculate_fitness(workers, jobs)
            fitness_time += time.time() - t0
            
            # Add to new population
            new_population.append(child)
            
            # Update archive
            archive.add(child)
        
        # Replace old population
        population = new_population
        
        # Record statistics
        valid_archive = [sol for sol in archive.solutions if sol.fitness_values is not None]
        valid_population = [ind for ind in population if ind.fitness_values is not None]
        
        if valid_archive and valid_population:
            best_shifts = min(sol.fitness_values[0] for sol in valid_archive)
            avg_shifts = sum(ind.fitness_values[0] for ind in valid_population) / len(valid_population)
            best_coverage = max(-sol.fitness_values[1] for sol in valid_archive)
            avg_coverage = sum(-ind.fitness_values[1] for ind in valid_population) / len(valid_population)
            best_overlap = min(sol.fitness_values[2] for sol in valid_archive)
            avg_overlap = sum(ind.fitness_values[2] for ind in valid_population) / len(valid_population)
            
            # Calculate more statistics for reporting
            archive_stats = archive.get_statistics()
            
            gen_time = time.time() - gen_start
            
            history.append({
                'generation': gen + 1,
                'best_shifts': best_shifts,
                'avg_shifts': avg_shifts,
                'best_coverage': best_coverage,
                'avg_coverage': avg_coverage,
                'best_overlap': best_overlap,
                'avg_overlap': avg_overlap,
                'archive_size': len(archive.solutions),
                'time': gen_time,
                'archive_stats': archive_stats
            })
            
            # Print progress
            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"Generation {gen+1}/{generations}, Archive size: {len(archive.solutions)}")
                print(f"Best fitness values: Shifts={best_shifts}, Coverage={best_coverage}/{len(jobs)}, Overlap={best_overlap}")
    
    # Calculate performance statistics
    total_time = time.time() - start_time
    stats = {
        'total_time': total_time,
        'repair_time': repair_time,
        'fitness_time': fitness_time,
        'selection_time': selection_time,
        'crossover_time': crossover_time,
        'mutation_time': mutation_time,
        'generations': generations,
        'population_size': population_size,
        'archive_size': len(archive.solutions),
        'params': {
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'tournament_size': tournament_size,
            'seed': seed
        }
    }
    
    return archive, history, stats

# Visualization functions
def visualize_pareto_front(archive):
    """Visualize the Pareto front for the first two objectives."""
    if not archive.solutions:
        print("No solutions to visualize Pareto front")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No solutions in archive", ha='center', va='center')
        # Ensure this line is inside a function. For example:
        def visualize_convergence(history):
            """
            Visualize the convergence of the algorithm.
            This function plots the best and average values of objectives over generations.
            """
            if not history:
                print("No history data to visualize convergence")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
                # Ensure this line is inside a function. For example:
                def visualize_convergence(history):
                    """
                    Visualize the convergence of the algorithm.
                    This function plots the best and average values of objectives over generations.
                    """
                    if not history:
                        print("No history data to visualize convergence")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
                        # Ensure this line is inside a function. For example:
                        def visualize_convergence(history):
                            """
                            Visualize the convergence of the algorithm.
                            This function plots the best and average values of objectives over generations.
                            """
                            if not history:
                                print("No history data to visualize convergence")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
                                return fig
        
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
            # Plot each objective
            generations = range(1, len(history) + 1)
        
            # Objective 1: Minimize shifts
            axs[0].plot(generations, [gen['best_shifts'] for gen in history], 'b-', label='Best')
            axs[0].plot(generations, [gen['avg_shifts'] for gen in history], 'r--', label='Average')
            axs[0].set_ylabel('Number of Shifts')
            axs[0].set_title('Convergence: Shifts (minimize)')
            axs[0].legend()
            axs[0].grid(True, linestyle='--', alpha=0.7)
        
            # Objective 2: Maximize coverage
            axs[1].plot(generations, [gen['best_coverage'] for gen in history], 'b-', label='Best')
            axs[1].plot(generations, [gen['avg_coverage'] for gen in history], 'r--', label='Average')
            axs[1].set_ylabel('Job Coverage')
            axs[1].set_title('Convergence: Coverage (maximize)')
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.7)
        
            # Objective 3: Minimize overlap
            axs[2].plot(generations, [gen['best_overlap'] for gen in history], 'b-', label='Best')
            axs[2].plot(generations, [gen['avg_overlap'] for gen in history], 'r--', label='Average')
            axs[2].set_xlabel('Generation')
            axs[2].set_ylabel('Overlap Penalty')
            axs[2].set_title('Convergence: Overlap (minimize)')
            axs[2].legend()
            axs[2].grid(True, linestyle='--', alpha=0.7)
        
            plt.tight_layout()
            plt.savefig('convergence.png')
            plt.close()
        
            return fig
            if not history:
                print("No history data to visualize convergence")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
                return fig
            # Add the rest of the function logic here
    
    # Extract objective values
    x_values = [sol.fitness_values[0] for sol in archive.solutions if sol.fitness_values]
    y_values = [-sol.fitness_values[1] for sol in archive.solutions if sol.fitness_values]  # Negate to show as maximize
    
    if not x_values or not y_values:
        print("No valid fitness values to visualize")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid fitness values", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Pareto front
    scatter = ax.scatter(x_values, y_values, c='blue', marker='o')
    
    # Connect points to show front
    points = sorted(zip(x_values, y_values))
    x_sorted, y_sorted = zip(*points)
    ax.plot(x_sorted, y_sorted, 'b--', alpha=0.5)
    
    # Add solution indices
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Number of Shifts (minimize)')
    ax.set_ylabel('Job Coverage (maximize)')
    ax.set_title('Pareto Front: Shifts vs Coverage')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('pareto_front.png')
    plt.close()
    
    return fig

def visualize_pareto_front_3d(archive):
    """Visualize the 3D Pareto front with all three objectives."""
    if not archive.solutions:
        print("No solutions to visualize 3D Pareto front")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No solutions in archive", ha='center', va='center')
        return fig
    
    # Extract objective values
    x_values = [sol.fitness_values[0] for sol in archive.solutions if sol.fitness_values]
    y_values = [-sol.fitness_values[1] for sol in archive.solutions if sol.fitness_values]  # Negate to show as maximize
    z_values = [sol.fitness_values[2] for sol in archive.solutions if sol.fitness_values]
    
    if not x_values or not y_values or not z_values:
        print("No valid fitness values to visualize")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No valid fitness values", ha='center', va='center')
        return fig
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto front
    scatter = ax.scatter(x_values, y_values, z_values, c='blue', marker='o')
    
    # Add solution indices
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        ax.text(x, y, z, str(i))
    
    ax.set_xlabel('Number of Shifts (minimize)')
    ax.set_ylabel('Job Coverage (maximize)')
    ax.set_zlabel('Overlap Penalty (minimize)')
    ax.set_title('3D Pareto Front')
    
    plt.savefig('pareto_front_3d.png')
    plt.close()
    
    return fig

def visualize_solution(solution, workers, jobs):
    """Visualize a solution as a colorful Gantt chart."""
    if not solution or solution.fitness_values is None:
        print("No valid solution to visualize")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid solution", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Collect assigned jobs for each worker
    worker_jobs = {}
    for w_idx in range(len(workers)):
        assigned_jobs = []
        for j_idx in range(len(jobs)):
            if solution.assignment_matrix[w_idx, j_idx] == 1:
                job = jobs[j_idx]
                assigned_jobs.append((job.id, job.start_time, job.end_time))
        
        if assigned_jobs:  # Only include workers with assigned jobs
            worker_jobs[f"Worker {w_idx}"] = assigned_jobs
    
    # Define bright colors for jobs
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.Set3(np.linspace(0, 1, 12))])
    
    # Plot timeline for each worker
    y_pos = 0
    yticks = []
    yticklabels = []
    
    for worker_name, jobs_list in worker_jobs.items():
        yticks.append(y_pos)
        yticklabels.append(worker_name)
        
        for job_id, start, end in jobs_list:
            color_idx = job_id % len(colors)
            
            bar = ax.barh(y_pos, end - start, left=start, height=0.5, 
                         color=colors[color_idx], edgecolor='white', linewidth=0.5)
            
            ax.text((start + end) / 2, y_pos, f"J{job_id}", 
                   ha='center', va='center', color='black',
                   fontsize=9, fontweight='bold')
        
        y_pos += 1
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_title('Job Assignments')
    
    # Add fitness info
    shifts = solution.fitness_values[0]
    coverage = -solution.fitness_values[1]
    overlap = solution.fitness_values[2]
    info_text = f"Shifts: {shifts}, Coverage: {coverage}/{len(jobs)}, Overlap: {overlap}"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('best_solution.png')
    plt.close()
    
    return fig

def visualize_convergence(history):
    """Visualize the convergence of the algorithm."""
    if not history:
        print("No history data to visualize convergence")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
        return fig
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot each objective
    generations = range(1, len(history) + 1)
    
    # Objective 1: Minimize shifts
    axs[0].plot(generations, [gen['best_shifts'] for gen in history], 'b-', label='Best')
    axs[0].plot(generations, [gen['avg_shifts'] for gen in history], 'r--', label='Average')
    axs[0].set_ylabel('Number of Shifts')
    axs[0].set_title('Convergence: Shifts (minimize)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Objective 2: Maximize coverage
    axs[1].plot(generations, [gen['best_coverage'] for gen in history], 'b-', label='Best')
    axs[1].plot(generations, [gen['avg_coverage'] for gen in history], 'r--', label='Average')
    axs[1].set_ylabel('Job Coverage')
    axs[1].set_title('Convergence: Coverage (maximize)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Objective 3: Minimize overlap
    axs[2].plot(generations, [gen['best_overlap'] for gen in history], 'b-', label='Best')
    axs[2].plot(generations, [gen['avg_overlap'] for gen in history], 'r--', label='Average')
    axs[2].set_xlabel('Generation')
    axs[2].set_ylabel('Overlap Penalty')
    axs[2].set_title('Convergence: Overlap (minimize)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('convergence.png')
    plt.close()
    
    return fig

def visualize_worker_assignment_heatmap(archive, workers, jobs):
   """Create a heatmap showing which workers are used across Pareto front solutions."""
   if not archive.solutions:
       print("No solutions to create worker assignment heatmap")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No solutions available", ha='center', va='center')
       return fig
   
   # Create a matrix: rows = solutions, columns = workers
   assignment_matrix = np.zeros((len(archive.solutions), len(workers)))
   
   # Fill the matrix with 1 if worker is used in solution, 0 otherwise
   for i, solution in enumerate(archive.solutions):
       if solution.fitness_values is None:
           continue
       for w_idx in range(len(workers)):
           if np.any(solution.assignment_matrix[w_idx]):
               assignment_matrix[i, w_idx] = 1
   
   # Calculate worker usage frequency
   worker_usage = np.sum(assignment_matrix, axis=0) / max(1, len(archive.solutions))
   
   # Create a heatmap
   fig, ax = plt.subplots(figsize=(12, 8))
   im = ax.imshow(assignment_matrix, aspect='auto', cmap='Greys', interpolation='none')
   
   # Add colorbar
   cbar = ax.figure.colorbar(im, ax=ax)
   cbar.ax.set_ylabel("Worker Used (1) or Not (0)", rotation=-90, va="bottom")
   
   # Set labels
   ax.set_xlabel("Worker ID")
   ax.set_ylabel("Solution Index")
   ax.set_title("Worker Assignment Across Pareto Front Solutions")
   
   # Set ticks
   ax.set_xticks(range(len(workers)))
   ax.set_xticklabels([f"W{i}" for i in range(len(workers))], rotation=90)
   ax.set_yticks(range(len(archive.solutions)))
   
   # Add a text annotation showing the worker usage percentage
   for w_idx in range(len(workers)):
       ax.text(w_idx, len(archive.solutions) + 0.5, f"{worker_usage[w_idx]:.1f}", 
              ha='center', va='center', rotation=90, fontsize=8)
   
   plt.tight_layout()
   plt.savefig('worker_assignment_heatmap.png')
   plt.close()
   
   return fig

def visualize_shifts_vs_penalties(archive):
   """Create a scatter plot of shifts vs overlap penalties in the Pareto front."""
   if not archive.solutions:
       print("No solutions to visualize shifts vs penalties")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No solutions available", ha='center', va='center')
       return fig
   
   # Extract objective values
   valid_solutions = [sol for sol in archive.solutions if sol.fitness_values is not None]
   if not valid_solutions:
       print("No valid fitness values to visualize")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No valid fitness values", ha='center', va='center')
       return fig
       
   shift_values = [sol.fitness_values[0] for sol in valid_solutions]
   penalty_values = [sol.fitness_values[2] for sol in valid_solutions]
   coverage_values = [-sol.fitness_values[1] for sol in valid_solutions]
   
   # Create scatter plot with points sized by coverage
   fig, ax = plt.subplots(figsize=(10, 6))
   scatter = ax.scatter(shift_values, penalty_values, 
                        c='black', edgecolors='grey',
                        s=100, alpha=0.7)
   
   # Set a professional-looking style
   plt.style.use('seaborn-v0_8-whitegrid')
   
   # Annotate each point with its solution index
   for i, (shifts, penalties) in enumerate(zip(shift_values, penalty_values)):
       ax.annotate(f"{i}", (shifts, penalties), 
                  xytext=(5, 5), textcoords='offset points')
   
   # Add a fit line to show the general trend
   if len(shift_values) > 1:
       z = np.polyfit(shift_values, penalty_values, 1)
       p = np.poly1d(z)
       ax.plot(sorted(shift_values), p(sorted(shift_values)), "r--", alpha=0.5)
   
   ax.set_xlabel('Number of Shifts (minimize)')
   ax.set_ylabel('Overlap Penalty (minimize)')
   ax.set_title('Pareto Front: Shifts vs Overlap Penalties')
   ax.grid(True, linestyle='--', alpha=0.7)
   
   plt.tight_layout()
   plt.savefig('shifts_vs_penalties.png')
   plt.close()
   
   return fig

def visualize_worker_load_distribution(solution, workers, jobs):
   """Visualize the workload distribution among workers."""
   if not solution or solution.fitness_values is None:
       print("No valid solution to visualize worker load")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No valid solution", ha='center', va='center')
       return fig
   
   worker_load = solution.get_worker_load(jobs)
   
   # Sort workers by load
   sorted_workers = sorted(worker_load.items(), key=lambda x: x[1], reverse=True)
   worker_ids = [f"Worker {w_id}" for w_id, _ in sorted_workers]
   loads = [load for _, load in sorted_workers]
   
   fig, ax = plt.subplots(figsize=(12, 6))
   bars = ax.bar(worker_ids, loads)
   
   # Add job count annotations
   for i, (w_id, _) in enumerate(sorted_workers):
       job_count = np.sum(solution.assignment_matrix[w_id])
       ax.text(i, loads[i] + 10, f"{int(job_count)} jobs", 
              ha='center', va='bottom')
   
   ax.set_xlabel('Worker')
   ax.set_ylabel('Total Working Time')
   ax.set_title('Worker Workload Distribution')
   ax.set_xticklabels(worker_ids, rotation=45, ha='right')
   
   # Add statistics
   mean_load = np.mean(loads)
   std_load = np.std(loads)
   ax.axhline(mean_load, color='r', linestyle='--', alpha=0.7)
   ax.text(len(worker_ids)-1, mean_load, f'Mean: {mean_load:.1f}', 
          va='bottom', ha='right', color='r')
   
   # Add fairness metric
   fairness = 1 - (std_load / mean_load if mean_load > 0 else 0)
   ax.text(0.02, 0.95, f'Fairness Index: {fairness:.2f}', 
          transform=ax.transAxes, fontsize=10,
          bbox=dict(facecolor='white', alpha=0.8))
   
   plt.tight_layout()
   plt.savefig('worker_load_distribution.png')
   plt.close()
   
   return fig

def visualize_algorithm_performance(stats):
   """Visualize the performance metrics of the algorithm."""
   if not stats:
       print("No performance stats to visualize")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No performance data available", ha='center', va='center')
       return fig
   
   # Create a pie chart of time distribution
   time_keys = ['repair_time', 'fitness_time', 'selection_time', 
               'crossover_time', 'mutation_time']
   time_labels = ['Repair', 'Fitness Evaluation', 'Selection', 
                 'Crossover', 'Mutation']
   
   time_values = [stats.get(key, 0) for key in time_keys]
   other_time = stats.get('total_time', sum(time_values)) - sum(time_values)
   
   time_values.append(other_time)
   time_labels.append('Other')
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.pie(time_values, labels=time_labels, autopct='%1.1f%%', 
         startangle=90, shadow=True)
   ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
   
   ax.set_title(f'Algorithm Time Distribution\nTotal Time: {stats.get("total_time", 0):.2f}s')
   
   plt.tight_layout()
   plt.savefig('algorithm_performance.png')
   plt.close()
   
   return fig

def visualize_pareto_evolution(history):
   """Visualize how the Pareto front evolves over generations."""
   if not history:
       print("No history data to visualize Pareto evolution")
       fig, ax = plt.subplots(figsize=(10, 6))
       ax.text(0.5, 0.5, "No evolution data available", ha='center', va='center')
       return fig
   
   # Select a subset of generations to visualize
   num_gens = len(history)
   if num_gens <= 5:
       selected_gens = list(range(num_gens))
   else:
       selected_gens = [0, num_gens//4, num_gens//2, 3*num_gens//4, num_gens-1]
   
   fig, ax = plt.subplots(figsize=(10, 6))
   
   colors = plt.cm.viridis(np.linspace(0, 1, len(selected_gens)))
   
   for i, gen_idx in enumerate(selected_gens):
       gen = history[gen_idx]
       ax.scatter(gen['best_shifts'], -gen['best_coverage'], 
                 color=colors[i], label=f"Gen {gen['generation']}")
   
   ax.set_xlabel('Number of Shifts (minimize)')
   ax.set_ylabel('Job Coverage (maximize)')
   ax.set_title('Evolution of Pareto Front')
   ax.grid(True, linestyle='--', alpha=0.7)
   ax.legend()
   
   plt.tight_layout()
   plt.savefig('pareto_evolution.png')
   plt.close()
   
   return fig

# Function to load the data
def load_data(file_path):
   """
   Load and parse data from the file.
   
   Args:
       file_path: Path to the data file
       
   Returns:
       tuple: (jobs, workers) lists of Job and Worker objects
   """
   try:
       data = pd.read_csv(file_path)
   except Exception as e:
       print(f"Error loading data file: {e}")
       return None, None
   
   # Extract the Jobs value
   jobs_match = re.search(r"Jobs = (\d+)", data.iloc[3, 0])
   if jobs_match:
       jobs_count = int(jobs_match.group(1))
       print(f"Extracted Jobs: {jobs_count}")
   else:
       print("Jobs not found.")
       return None, None
   
   # Extract the Qualifications value
   qualifications_match = re.search(r"Qualifications = (\d+)", data.iloc[jobs_count + 4, 0])
   if qualifications_match:
       qualifications_count = int(qualifications_match.group(1))
       print(f"Extracted Qualifications: {qualifications_count}")
   else:
       print("Qualifications not found.")
       return None, None
   
   # Extract the start and end times from the data
   start_end_data = data.iloc[4:4+jobs_count, 0]
   start_end_list = [list(map(int, row.split())) for row in start_end_data]
   
   # Extract lines containing qualified jobs
   qualified_jobs_lines = data.iloc[jobs_count + 5:, 0]
   qualified_jobs_pattern = re.compile(r"^\s*\d+:((?:\s+\d+)+)")
   
   qualified_jobs = []
   for line in qualified_jobs_lines:
       match = qualified_jobs_pattern.match(line)
       if match:
           jobs_list = list(map(int, match.group(1).split()))
           qualified_jobs.append(jobs_list)
   
   # Create job and worker objects
   jobs = [Job(i, start_end[0], start_end[1]) for i, start_end in enumerate(start_end_list)]
   workers = [Worker(i, quals) for i, quals in enumerate(qualified_jobs)]
   
   print(f"Created {len(jobs)} jobs and {len(workers)} workers")
   
   return jobs, workers

def run_experiments(file_path, seeds=None, params=None):
   """
   Run multiple experiments with different seeds and parameters.
   
   Args:
       file_path: Path to the data file
       seeds: List of random seeds to use
       params: List of parameter dictionaries
       
   Returns:
       list: Results from each experiment
   """
   if seeds is None:
       seeds = [42]
   
   if params is None:
       params = [{}]  # Default parameters
   
   results = []
   
   # Load data once
   jobs, workers = load_data(file_path)
   if not jobs or not workers:
       return results
   
   for seed in seeds:
       for param_set in params:
           print(f"\nRunning experiment with seed {seed} and parameters:")
           for k, v in param_set.items():
               print(f"  {k}: {v}")
           
           # Set experiment parameters
           population_size = param_set.get('population_size', 100)
           generations = param_set.get('generations', 50)
           archive_size = param_set.get('archive_size', 50)
           crossover_rate = param_set.get('crossover_rate', 0.8)
           mutation_rate = param_set.get('mutation_rate', 0.1)
           tournament_size = param_set.get('tournament_size', 3)
           
           try:
               # Run MOGA
               archive, history, stats = run_moga(
                   workers, jobs, 
                   population_size=population_size,
                   generations=generations,
                   archive_size=archive_size,
                   crossover_rate=crossover_rate,
                   mutation_rate=mutation_rate,
                   tournament_size=tournament_size,
                   seed=seed
               )
               
               # Get best solution for each objective
               valid_solutions = [sol for sol in archive.solutions if sol.fitness_values is not None]
               
               # Store results
               experiment_result = {
                   'seed': seed,
                   'params': param_set,
                   'stats': stats,
                   'archive_size': len(archive.solutions),
                   'valid_solutions': len(valid_solutions),
                   'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               }
               
               if valid_solutions:
                   # Get best solution for each objective
                   best_shifts_sol = min(valid_solutions, key=lambda x: x.fitness_values[0])
                   best_coverage_sol = max(valid_solutions, key=lambda x: -x.fitness_values[1])
                   best_overlap_sol = min(valid_solutions, key=lambda x: x.fitness_values[2])
                   
                   experiment_result.update({
                       'best_shifts': best_shifts_sol.fitness_values[0],
                       'best_coverage': -best_coverage_sol.fitness_values[1],
                       'best_overlap': best_overlap_sol.fitness_values[2],
                   })
                   
                   # Create visualizations for this experiment
                   visualize_solution(best_shifts_sol, workers, jobs)
                   visualize_pareto_front(archive)
                   visualize_pareto_front_3d(archive)
                   visualize_convergence(history)
                   visualize_worker_assignment_heatmap(archive, workers, jobs)
                   visualize_shifts_vs_penalties(archive)
                   visualize_worker_load_distribution(best_shifts_sol, workers, jobs)
                   visualize_algorithm_performance(stats)
                   visualize_pareto_evolution(history)
               
               results.append(experiment_result)
               
               print(f"Experiment completed successfully.")
               if valid_solutions:
                   print(f"Best shifts: {experiment_result.get('best_shifts')}")
                   print(f"Best coverage: {experiment_result.get('best_coverage')}/{len(jobs)}")
                   print(f"Best overlap: {experiment_result.get('best_overlap')}")
               
           except Exception as e:
               print(f"Error in experiment: {e}")
   
   return results

def compare_experiment_results(results):
   """
   Create visualizations comparing multiple experiment results.
   
   Args:
       results: List of experiment result dictionaries
       
   Returns:
       dict: Paths to the comparison visualizations
   """
   if not results:
       print("No experiment results to compare")
       return {}
   
   # Extract key metrics
   seeds = [r['seed'] for r in results]
   shifts = [r.get('best_shifts', float('inf')) for r in results]
   coverage = [r.get('best_coverage', 0) for r in results]
   overlaps = [r.get('best_overlap', float('inf')) for r in results]
   times = [r['stats'].get('total_time', 0) for r in results]
   
   # Create bar chart comparing results
   fig, axs = plt.subplots(2, 2, figsize=(12, 10))
   
   # Plot shifts
   axs[0, 0].bar(range(len(results)), shifts)
   axs[0, 0].set_xticks(range(len(results)))
   axs[0, 0].set_xticklabels([f"Exp {i}" for i in range(len(results))], rotation=45)
   axs[0, 0].set_ylabel('Number of Shifts')
   axs[0, 0].set_title('Best Shifts by Experiment')
   
   # Plot coverage
   axs[0, 1].bar(range(len(results)), coverage)
   axs[0, 1].set_xticks(range(len(results)))
   axs[0, 1].set_xticklabels([f"Exp {i}" for i in range(len(results))], rotation=45)
   axs[0, 1].set_ylabel('Job Coverage')
   axs[0, 1].set_title('Best Coverage by Experiment')
   
   # Plot overlaps
   axs[1, 0].bar(range(len(results)), overlaps)
   axs[1, 0].set_xticks(range(len(results)))
   axs[1, 0].set_xticklabels([f"Exp {i}" for i in range(len(results))], rotation=45)
   axs[1, 0].set_ylabel('Overlap Penalty')
   axs[1, 0].set_title('Best Overlap by Experiment')
   
   # Plot times
   axs[1, 1].bar(range(len(results)), times)
   axs[1, 1].set_xticks(range(len(results)))
   axs[1, 1].set_xticklabels([f"Exp {i}" for i in range(len(results))], rotation=45)
   axs[1, 1].set_ylabel('Execution Time (s)')
   axs[1, 1].set_title('Execution Time by Experiment')
   
   plt.tight_layout()
   plt.savefig('experiment_comparison.png')
   plt.close()
   
   # Create summary table
   summary = {
       'Best Shifts': min(shifts),
       'Best Coverage': max(coverage),
       'Best Overlap': min(overlaps),
       'Average Shifts': sum(shifts) / len(shifts),
       'Average Coverage': sum(coverage) / len(coverage),
       'Average Overlap': sum(overlaps) / len(overlaps),
       'Total Experiments': len(results)
   }
   
   return {
       'comparison_chart': 'experiment_comparison.png',
       'summary': summary
   }

# Main function to run the algorithm
def main():
   """Main entry point for the program."""
   # Set fixed random seed for reproducibility
   random.seed(RANDOM_SEED)
   np.random.seed(RANDOM_SEED)
   
   # File path
   file_path = "ptask/data_3_25_40_66.dat"
   
   print(f"Starting MOGA optimization for shift scheduling problem")
   print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   print(f"Using random seed: {RANDOM_SEED}")
   
   # Run single experiment with detailed output for report
   jobs, workers = load_data(file_path)
   if not jobs or not workers:
       print("Failed to load data.")
       return
   
   print(f"Loaded {len(jobs)} jobs and {len(workers)} workers.")
   
   # Run MOGA with detailed metrics
   archive, history, stats = run_moga(
       workers, jobs, 
       population_size=100,
       generations=50,
       archive_size=50,
       seed=RANDOM_SEED
   )
   
   # Get best solution for minimizing shifts
   valid_solutions = [sol for sol in archive.solutions if sol.fitness_values is not None]
   if valid_solutions:
       best_solution = min(valid_solutions, key=lambda x: x.fitness_values[0])
       print("\nBest solution for minimizing shifts:")
       print(f"Shifts: {best_solution.fitness_values[0]}")
       print(f"Coverage: {-best_solution.fitness_values[1]}/{len(jobs)}")
       print(f"Overlap penalty: {best_solution.fitness_values[2]}")
       
       # Create visualizations
       print("\nGenerating visualizations...")
       visualize_pareto_front(archive)
       visualize_pareto_front_3d(archive)
       visualize_solution(best_solution, workers, jobs)
       visualize_convergence(history)
       visualize_worker_assignment_heatmap(archive, workers, jobs)
       visualize_shifts_vs_penalties(archive)
       visualize_worker_load_distribution(best_solution, workers, jobs)
       visualize_algorithm_performance(stats)
       visualize_pareto_evolution(history)
       
       print("All visualizations saved to files.")
       
       # Print performance statistics
       print("\nPerformance Statistics:")
       print(f"Total execution time: {stats['total_time']:.2f}s")
       print(f"Time spent on repair operations: {stats['repair_time']:.2f}s ({100*stats['repair_time']/stats['total_time']:.1f}%)")
       print(f"Time spent on fitness evaluations: {stats['fitness_time']:.2f}s ({100*stats['fitness_time']/stats['total_time']:.1f}%)")
       print(f"Archive size: {len(archive.solutions)}")
       
       print("\nOptimization completed successfully.")
   else:
       print("\nNo valid solutions found.")

if __name__ == "__main__":
   main()