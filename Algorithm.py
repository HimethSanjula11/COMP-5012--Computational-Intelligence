import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import re
from typing import List, Dict, Tuple, Set
import pandas as pd
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class Job:
    def __init__(self, id, start_time, end_time):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
    
    def __str__(self):
        return f"Job {self.id}: [{self.start_time}, {self.end_time}], duration: {self.duration}"
    
    def overlaps(self, other_job):
        """Check if this job overlaps with another job."""
        return (self.start_time < other_job.end_time and
                self.end_time > other_job.start_time)

class Worker:
    def __init__(self, id, qualifications):
        self.id = id
        self.qualifications = qualifications  # List of job types this worker can perform
    
    def __str__(self):
        return f"Worker {self.id}: qualified for jobs {self.qualifications}"
    
    def can_perform(self, job_type):
        """Check if the worker can perform a job of given type."""
        return job_type in self.qualifications

class ShiftSchedulingProblem:
    def __init__(self, data_file):
        self.jobs = []
        self.workers = []
        self.parse_data(data_file)
        self.worker_count = len(self.workers)
        self.job_count = len(self.jobs)
    
    def parse_data(self, data_file):
        """Parse the problem data from a file."""
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines
        line_idx = 0
        while line_idx < len(lines) and lines[line_idx].startswith('#'):
            line_idx += 1
        
        # Parse problem type
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith('Type'):
                self.problem_type = int(line.split('=')[1].strip())
                line_idx += 1
                break
            line_idx += 1
        
        # Parse number of jobs
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith('Jobs'):
                num_jobs = int(line.split('=')[1].strip())
                line_idx += 1
                break
            line_idx += 1
        
        # Parse job data
        for i in range(num_jobs):
            if line_idx < len(lines):
                line = lines[line_idx].strip()
                start_time, end_time = map(int, line.split())
                self.jobs.append(Job(i, start_time, end_time))
                line_idx += 1
        
        # Parse number of qualifications
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith('Qualifications'):
                num_qualifications = int(line.split('=')[1].strip())
                line_idx += 1
                break
            line_idx += 1
        
        # Parse worker qualifications
        for i in range(num_qualifications):
            if line_idx < len(lines):
                line = lines[line_idx].strip()
                parts = line.split(':')
                if len(parts) > 1:
                    quals = list(map(int, parts[1].strip().split()))
                    self.workers.append(Worker(i, quals))
                line_idx += 1
    
    def visualize_jobs(self):
        """Visualize jobs as a timeline."""
        fig, ax = plt.subplots(figsize=(15, 8))
        for i, job in enumerate(self.jobs):
            ax.barh(i, job.duration, left=job.start_time, height=0.5, 
                   align='center', color='blue', alpha=0.6)
            ax.text(job.start_time + job.duration/2, i, f"Job {job.id}", 
                   ha='center', va='center', color='white')
        
        ax.set_yticks(range(len(self.jobs)))
        ax.set_yticklabels([f"Job {job.id}" for job in self.jobs])
        ax.set_xlabel('Time')
        ax.set_title('Job Timeline')
        plt.tight_layout()
        return fig

class Individual:
    def __init__(self, assignment_matrix=None, problem=None):
        self.problem = problem
        if assignment_matrix is not None:
            self.assignment_matrix = assignment_matrix
        elif problem is not None:
            # Initialize randomly
            self.assignment_matrix = np.zeros((problem.worker_count, problem.job_count), dtype=int)
            for j in range(problem.job_count):
                # Find qualified workers for this job
                qualified_workers = [w.id for w in problem.workers if j in w.qualifications]
                if qualified_workers:
                    # Assign job to a random qualified worker
                    worker_id = random.choice(qualified_workers)
                    self.assignment_matrix[worker_id, j] = 1
        
        self.fitness_values = None
    
    def calculate_fitness(self):
        """Calculate fitness values (multiple objectives)."""
        if self.problem is None:
            raise ValueError("Problem not set for individual")
        
        # Check if solution is valid (all jobs assigned to qualified workers)
        is_valid = True
        for j_idx in range(self.problem.job_count):
            assigned = False
            for w_idx in range(self.problem.worker_count):
                if self.assignment_matrix[w_idx, j_idx] == 1:
                    if j_idx not in self.problem.workers[w_idx].qualifications:
                        is_valid = False
                        break
                    assigned = True
            if not assigned:
                is_valid = False
                break
        
        if not is_valid:
            self.fitness_values = None
            return None
            
        # Objective 1: Minimize number of shifts (workers that are assigned at least one job)
        shifts_used = np.sum(np.any(self.assignment_matrix, axis=1))
        
        # Objective 2: Maximize job coverage (jobs that are assigned to a worker)
        jobs_covered = np.sum(np.any(self.assignment_matrix, axis=0))
        
        # Objective 3: Minimize overlapping jobs per worker
        overlap_penalty = 0
        for w in range(self.problem.worker_count):
            assigned_jobs = [j for j in range(self.problem.job_count) if self.assignment_matrix[w, j] == 1]
            for i, j1_idx in enumerate(assigned_jobs):
                for j2_idx in assigned_jobs[i+1:]:
                    if self.problem.jobs[j1_idx].overlaps(self.problem.jobs[j2_idx]):
                        overlap_penalty += 1
        
        # Store and return fitness values (note: we minimize objective 1 and 3, maximize objective 2)
        self.fitness_values = [shifts_used, -jobs_covered, overlap_penalty]
        return self.fitness_values
    
    def is_valid(self):
        """Check if this is a valid solution (all jobs assigned to qualified workers, no overlaps)."""
        # Check if all jobs are assigned to at least one worker
        if not np.all(np.sum(self.assignment_matrix, axis=0) >= 1):
            return False
            
        # Check if jobs are assigned to qualified workers
        for w_idx in range(self.problem.worker_count):
            for j_idx in range(self.problem.job_count):
                if self.assignment_matrix[w_idx, j_idx] == 1:
                    if j_idx not in self.problem.workers[w_idx].qualifications:
                        return False
        
        # Check for overlapping jobs for each worker
        for w_idx in range(self.problem.worker_count):
            assigned_jobs = [j for j in range(self.problem.job_count) if self.assignment_matrix[w_idx, j] == 1]
            for i, j1_idx in enumerate(assigned_jobs):
                for j2_idx in assigned_jobs[i+1:]:
                    if self.problem.jobs[j1_idx].overlaps(self.problem.jobs[j2_idx]):
                        return False
        
        return True
    
    def repair(self):
        """Repair an invalid solution by removing conflicting assignments."""
        # First, check if all jobs are assigned to qualified workers
        for w_idx in range(self.problem.worker_count):
            for j_idx in range(self.problem.job_count):
                if (self.assignment_matrix[w_idx, j_idx] == 1 and 
                    j_idx not in self.problem.workers[w_idx].qualifications):
                    self.assignment_matrix[w_idx, j_idx] = 0
        
        # Then, resolve overlapping jobs for each worker
        for w_idx in range(self.problem.worker_count):
            assigned_jobs = [(j_idx, self.problem.jobs[j_idx]) 
                             for j_idx in range(self.problem.job_count) 
                             if self.assignment_matrix[w_idx, j_idx] == 1]
            
            # Sort jobs by end time
            assigned_jobs.sort(key=lambda x: x[1].end_time)
            
            # Greedy algorithm to select non-overlapping jobs
            selected = []
            for j_idx, job in assigned_jobs:
                if not any(job.overlaps(selected_job[1]) for selected_job in selected):
                    selected.append((j_idx, job))
                else:
                    self.assignment_matrix[w_idx, j_idx] = 0
        
        # Finally, ensure all jobs are assigned
        for j_idx in range(self.problem.job_count):
            if np.sum(self.assignment_matrix[:, j_idx]) == 0:
                # Find qualified workers for this job
                qualified_workers = [w.id for w in self.problem.workers if j_idx in w.qualifications]
                
                # Try to assign to a worker with fewest conflicts
                best_worker = None
                min_conflicts = float('inf')
                
                for w_idx in qualified_workers:
                    assigned_jobs = [self.problem.jobs[j] for j in range(self.problem.job_count) 
                                     if self.assignment_matrix[w_idx, j] == 1]
                    conflicts = sum(1 for job in assigned_jobs 
                                    if job.overlaps(self.problem.jobs[j_idx]))
                    
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_worker = w_idx
                
                if best_worker is not None:
                    self.assignment_matrix[best_worker, j_idx] = 1
    
    def clone(self):
        """Create a copy of this individual."""
        new_individual = Individual(problem=self.problem)
        new_individual.assignment_matrix = self.assignment_matrix.copy()
        return new_individual

def dominates(fitness1, fitness2):
    """Check if fitness1 dominates fitness2."""
    # Check if either fitness is None
    if fitness1 is None or fitness2 is None:
        return False
        
    better_in_at_least_one = False
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:  # For objectives to maximize
            return False
        if f1 < f2:  # For objectives to minimize
            better_in_at_least_one = True
    return better_in_at_least_one

class Archive:
    def __init__(self, max_size=100):
        self.solutions = []
        self.max_size = max_size
    
    def add(self, individual):
        """Add an individual to the archive if it's non-dominated."""
        # Skip individuals with None fitness values
        if individual.fitness_values is None:
            return False
            
        # Check if individual is dominated by any in archive
        for sol in self.solutions:
            if sol.fitness_values is not None and dominates(sol.fitness_values, individual.fitness_values):
                return False
        
        # Remove solutions that are dominated by the new individual
        self.solutions = [sol for sol in self.solutions 
                         if sol.fitness_values is None or 
                         not dominates(individual.fitness_values, sol.fitness_values)]
        
        # Add new individual
        self.solutions.append(individual.clone())
        
        # If archive exceeds max size, perform pruning
        if len(self.solutions) > self.max_size:
            self._prune_archive()
        
        return True
    
    def _prune_archive(self):
        """Reduce archive size by removing solutions with similar fitness values."""
        # Calculate crowding distance
        distances = self._calculate_crowding_distance()
        
        # Sort solutions by crowding distance (higher distance = more unique)
        sorted_indices = np.argsort(distances)
        
        # Keep only max_size solutions with highest crowding distance
        self.solutions = [self.solutions[i] for i in sorted_indices[-self.max_size:]]
    
    def _calculate_crowding_distance(self):
        """Calculate crowding distance for each solution in archive."""
        num_solutions = len(self.solutions)
        if num_solutions <= 2:
            return [float('inf')] * num_solutions
            
        # Filter out solutions with None fitness values
        valid_solutions = [sol for sol in self.solutions if sol.fitness_values is not None]
        if not valid_solutions:
            return [0] * num_solutions
            
        num_objectives = len(valid_solutions[0].fitness_values)
        distances = np.zeros(num_solutions)
        
        # Create mapping between original indices and valid solution indices
        valid_indices = {i: j for j, i in enumerate([self.solutions.index(sol) for sol in valid_solutions])}
        
        for obj in range(num_objectives):
            try:
                # Extract fitness values for this objective (only for valid solutions)
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
    
    def get_random_solution(self):
        """Return a random solution from the archive."""
        if not self.solutions:
            return None
        return random.choice(self.solutions).clone()
    
    def get_best_solution(self, objective_idx=0):
        """Return the best solution for a specific objective."""
        if not self.solutions:
            return None
            
        # Filter out solutions with None fitness values
        valid_solutions = [sol for sol in self.solutions if sol.fitness_values is not None]
        
        if not valid_solutions:
            return None
            
        return min(valid_solutions, key=lambda x: x.fitness_values[objective_idx]).clone()

class MOGA:
    def __init__(self, problem, population_size=100, archive_size=50):
        self.problem = problem
        self.population_size = population_size
        self.population = []
        self.archive = Archive(max_size=archive_size)
        self.generation = 0
        self.history = {
            'avg_fitness': [],
            'best_fitness': [],
            'pareto_front': []
        }
    
    def initialize_population(self):
        """Initialize a random population."""
        self.population = []
        for _ in range(self.population_size):
            ind = Individual(problem=self.problem)
            ind.repair()  # Ensure solution is valid
            # Calculate fitness before adding to population or archive
            fitness = ind.calculate_fitness()
            if fitness is None:
                continue  # Skip individuals with invalid fitness
                
            self.population.append(ind)
            self.archive.add(ind)
        
        # Ensure we have enough individuals in the population
        while len(self.population) < self.population_size:
            ind = Individual(problem=self.problem)
            ind.repair()
            fitness = ind.calculate_fitness()
            if fitness is not None:
                self.population.append(ind)
                self.archive.add(ind)
                
        self._update_history()
    
    def _update_history(self):
        """Update history with current generation data."""
        if not self.population:
            return
            
        # Only consider individuals with valid fitness values
        valid_population = [ind for ind in self.population if ind.fitness_values is not None]
        if not valid_population:
            return
            
        # Calculate average fitness for each objective
        avg_fitness = [0] * len(valid_population[0].fitness_values)
        for ind in valid_population:
            for i, fitness in enumerate(ind.fitness_values):
                avg_fitness[i] += fitness / len(valid_population)
        
        # Get best fitness for each objective
        best_fitness = []
        for obj_idx in range(len(valid_population[0].fitness_values)):
            best_val = min(ind.fitness_values[obj_idx] for ind in valid_population)
            best_fitness.append(best_val)
        
        # Store current Pareto front
        pareto_front = []
        for sol in self.archive.solutions:
            if sol.fitness_values is not None:  # Skip solutions with None fitness
                pareto_front.append(sol.fitness_values.copy())
        
        # Update history
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_fitness'].append(best_fitness)
        self.history['pareto_front'].append(pareto_front)
    
    def tournament_selection(self, tournament_size=3):
        """Select an individual using tournament selection."""
        # Filter out individuals with None fitness values
        valid_population = [ind for ind in self.population if ind.fitness_values is not None]
        
        if not valid_population:
            # If no valid individuals, create a new random one
            ind = Individual(problem=self.problem)
            ind.repair()
            ind.calculate_fitness()
            return ind
            
        # Select candidates for tournament
        candidates = random.sample(valid_population, min(tournament_size, len(valid_population)))
        if not candidates:
            # Fallback if still no valid candidates
            ind = Individual(problem=self.problem)
            ind.repair()
            ind.calculate_fitness()
            return ind
            
        # Select the candidate that dominates the most others
        best_candidate = candidates[0]
        best_domination_count = 0
        
        for candidate in candidates:
            domination_count = sum(1 for other in candidates 
                                  if other.fitness_values is not None and 
                                  candidate.fitness_values is not None and
                                  dominates(candidate.fitness_values, other.fitness_values))
            
            if domination_count > best_domination_count:
                best_domination_count = domination_count
                best_candidate = candidate
        
        return best_candidate
    
    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """Perform crossover between two parents."""
        if random.random() > crossover_rate:
            return parent1.clone()
        
        # Uniform crossover
        child_matrix = np.zeros_like(parent1.assignment_matrix)
        
        for w in range(self.problem.worker_count):
            for j in range(self.problem.job_count):
                # Randomly choose assignment from either parent
                if random.random() < 0.5:
                    child_matrix[w, j] = parent1.assignment_matrix[w, j]
                else:
                    child_matrix[w, j] = parent2.assignment_matrix[w, j]
        
        child = Individual(assignment_matrix=child_matrix, problem=self.problem)
        return child
    
    def mutation(self, individual, mutation_rate=0.1):
        """Perform mutation on an individual."""
        for j in range(self.problem.job_count):
            if random.random() < mutation_rate:
                # For this job, reassign it to a random qualified worker
                qualified_workers = [w.id for w in self.problem.workers if j in w.qualifications]
                if qualified_workers:
                    # Remove current assignment
                    for w in range(self.problem.worker_count):
                        individual.assignment_matrix[w, j] = 0
                    
                    # Make new assignment
                    new_worker = random.choice(qualified_workers)
                    individual.assignment_matrix[new_worker, j] = 1
        
        return individual
    
    def evolve(self, generations=100):
        """Run the MOGA for a specified number of generations."""
        self.initialize_population()
        
        for gen in range(generations):
            self.generation = gen + 1
            
            new_population = []
            
            # Elitism: Include some solutions directly from archive
            elite_count = min(10, len(self.archive.solutions))
            if elite_count > 0:
                # Filter out solutions with None fitness values
                valid_elites = [elite for elite in self.archive.solutions if elite.fitness_values is not None]
                if valid_elites:
                    elites = random.sample(valid_elites, min(elite_count, len(valid_elites)))
                    new_population.extend([elite.clone() for elite in elites])
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                try:
                    # Selection
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                    
                    # Crossover
                    child = self.crossover(parent1, parent2)
                    
                    # Mutation
                    child = self.mutation(child)
                    
                    # Repair if needed
                    child.repair()
                    
                    # Calculate fitness
                    fitness = child.calculate_fitness()
                    
                    # Only add valid solutions
                    if fitness is not None:
                        # Add to new population
                        new_population.append(child)
                        
                        # Update archive
                        self.archive.add(child)
                except Exception as e:
                    print(f"Error during evolution: {e}")
                    continue
            
            # Replace old population
            self.population = new_population
            
            # Update history
            self._update_history()
            
            # Print progress
            if (gen + 1) % 10 == 0 or gen == 0:
                # Filter out individuals with None fitness
                valid_population = [ind for ind in self.population if ind.fitness_values is not None]
                
                if valid_population:
                    best_f1 = min(ind.fitness_values[0] for ind in valid_population)
                    best_f2 = min(ind.fitness_values[1] for ind in valid_population)
                    best_f3 = min(ind.fitness_values[2] for ind in valid_population)
                    print(f"Generation {gen+1}/{generations}, Archive size: {len(self.archive.solutions)}")
                    print(f"Best fitness values: Shifts={best_f1}, Coverage={-best_f2}, Overlap={best_f3}")
                else:
                    print(f"Generation {gen+1}/{generations}, No valid solutions in population")
    
    def visualize_pareto_front(self):
        """Visualize the Pareto front for the first two objectives."""
        try:
            if not self.archive.solutions:
                print("No solutions in archive to visualize Pareto front")
                # Create a simple figure with a message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No solutions in archive to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_xlabel('Number of Shifts (minimize)')
                ax.set_ylabel('Job Coverage (maximize)')
                ax.set_title('Pareto Front: Shifts vs Coverage (No valid solutions)')
                return fig
                
            # Filter solutions with valid fitness values
            valid_solutions = [sol for sol in self.archive.solutions if sol.fitness_values is not None]
            
            if not valid_solutions:
                print("No valid solutions to visualize Pareto front")
                # Create a simple figure with a message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No valid solutions to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_xlabel('Number of Shifts (minimize)')
                ax.set_ylabel('Job Coverage (maximize)')
                ax.set_title('Pareto Front: Shifts vs Coverage (No valid solutions)')
                return fig
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract objective values
            x_values = [sol.fitness_values[0] for sol in valid_solutions]
            y_values = [-sol.fitness_values[1] for sol in valid_solutions]  # Negate to show as maximize
            
            # Plot Pareto front
            ax.scatter(x_values, y_values, c='blue', marker='o')
            
            # Connect points to show front
            points = list(zip(x_values, y_values))
            if points:
                points.sort()
                x_sorted, y_sorted = zip(*points)
                ax.plot(x_sorted, y_sorted, 'b--', alpha=0.5)
            
            ax.set_xlabel('Number of Shifts (minimize)')
            ax.set_ylabel('Job Coverage (maximize)')
            ax.set_title('Pareto Front: Shifts vs Coverage')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            return fig
        except Exception as e:
            print(f"Error in visualize_pareto_front: {e}")
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error visualizing Pareto front: {e}", 
                  ha='center', va='center', fontsize=12, wrap=True)
            return fig
    
    def visualize_pareto_front_3d(self):
        """Visualize the Pareto front for all three objectives in 3D."""
        try:
            if not self.archive.solutions:
                print("No solutions in archive to visualize 3D Pareto front")
                # Create a simple figure with a message
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No solutions in archive to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_title('3D Pareto Front (No valid solutions)')
                return fig
                
            # Filter solutions with valid fitness values
            valid_solutions = [sol for sol in self.archive.solutions if sol.fitness_values is not None]
            
            if not valid_solutions:
                print("No valid solutions to visualize 3D Pareto front")
                # Create a simple figure with a message
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No valid solutions to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_title('3D Pareto Front (No valid solutions)')
                return fig
                
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract objective values
            x_values = [sol.fitness_values[0] for sol in valid_solutions]
            y_values = [-sol.fitness_values[1] for sol in valid_solutions]  # Negate to show as maximize
            z_values = [sol.fitness_values[2] for sol in valid_solutions]
            
            # Plot Pareto front
            ax.scatter(x_values, y_values, z_values, c='blue', marker='o')
            
            ax.set_xlabel('Number of Shifts (minimize)')
            ax.set_ylabel('Job Coverage (maximize)')
            ax.set_zlabel('Overlap Penalty (minimize)')
            ax.set_title('3D Pareto Front')
            
            return fig
        except Exception as e:
            print(f"Error in visualize_pareto_front_3d: {e}")
            # Create a simple error figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error visualizing 3D Pareto front: {e}", 
                  ha='center', va='center', fontsize=12, wrap=True)
            return fig
    
    def visualize_convergence(self):
        """Visualize the convergence of the algorithm."""
        try:
            if not self.history['best_fitness']:
                print("No convergence history to visualize")
                # Create a simple figure with a message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No convergence history to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_title('Convergence (No data)')
                return fig
                
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
            
            generations = range(1, len(self.history['best_fitness']) + 1)
            
            # Plot convergence for each objective
            obj_names = ['Shifts Used', 'Job Coverage (negated)', 'Overlap Penalty']
            
            for i in range(3):
                try:
                    axs[i].plot(generations, [bf[i] for bf in self.history['best_fitness']], 'b-', label='Best')
                    axs[i].plot(generations, [af[i] for af in self.history['avg_fitness']], 'r--', label='Average')
                    axs[i].set_xlabel('Generation')
                    axs[i].set_ylabel(obj_names[i])
                    axs[i].set_title(f'Convergence: {obj_names[i]}')
                    axs[i].legend()
                    axs[i].grid(True, linestyle='--', alpha=0.7)
                except Exception as e:
                    print(f"Error plotting objective {i}: {e}")
                    axs[i].text(0.5, 0.5, f"Error plotting data: {e}", 
                              ha='center', va='center', fontsize=10, transform=axs[i].transAxes)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in visualize_convergence: {e}")
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error visualizing convergence: {e}", 
                  ha='center', va='center', fontsize=12, wrap=True)
            return fig
    
    def visualize_solution(self, solution=None):
        """Visualize a specific solution as a Gantt chart."""
        try:
            if solution is None and self.archive.solutions:
                # Use solution with minimum shifts
                solution = self.archive.get_best_solution(objective_idx=0)
            
            if solution is None or solution.fitness_values is None:
                print("No valid solution to visualize")
                # Create a simple figure with a message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No valid solution to visualize", 
                      ha='center', va='center', fontsize=14)
                ax.set_title('Solution Visualization (No valid solution)')
                return fig
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Collect assigned jobs for each worker
            worker_jobs = {}
            for w_idx in range(self.problem.worker_count):
                assigned_jobs = []
                for j_idx in range(self.problem.job_count):
                    if solution.assignment_matrix[w_idx, j_idx] == 1:
                        job = self.problem.jobs[j_idx]
                        assigned_jobs.append((job.id, job.start_time, job.end_time))
                
                if assigned_jobs:  # Only include workers with assigned jobs
                    worker_jobs[f"Worker {w_idx}"] = assigned_jobs
            
            # Plot timeline for each worker
            y_pos = 0
            yticks = []
            yticklabels = []
            
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            
            for worker_name, jobs in worker_jobs.items():
                yticks.append(y_pos)
                yticklabels.append(worker_name)
                
                for job_id, start, end in jobs:
                    color_idx = job_id % len(colors)
                    ax.barh(y_pos, end - start, left=start, height=0.5, 
                           align='center', color=colors[color_idx], alpha=0.8)
                    ax.text((start + end) / 2, y_pos, f"J{job_id}", 
                           ha='center', va='center', color='black')
                
                y_pos += 1
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_xlabel('Time')
            ax.set_title('Solution: Job Assignments')
            ax.grid(True, linestyle='--', alpha=0.3, axis='x')
            
            # Add information about the solution
            solution.calculate_fitness()
            text_info = (f"Solution metrics:\n"
                         f"- Number of shifts: {solution.fitness_values[0]}\n"
                         f"- Jobs covered: {-solution.fitness_values[1]}/{self.problem.job_count}\n"
                         f"- Overlap penalty: {solution.fitness_values[2]}")
            
            ax.text(0.02, 0.02, text_info, transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in visualize_solution: {e}")
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error visualizing solution: {e}", 
                  ha='center', va='center', fontsize=12, wrap=True)
            return fig

# Main function to run the MOGA
def run_shift_scheduling_moga(data_file, generations=100, population_size=100, archive_size=50):
    """Run the MOGA for shift scheduling and return results."""
    try:
        problem = ShiftSchedulingProblem(data_file)
        
        # Show problem information
        print(f"Problem loaded: {len(problem.jobs)} jobs, {len(problem.workers)} workers")
        
        # Run MOGA
        moga = MOGA(problem, population_size=population_size, archive_size=archive_size)
        moga.evolve(generations=generations)
        
        return moga
    except Exception as e:
        print(f"Error in MOGA execution: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Get absolute path to the data file in the ptask folder
    import os
    # Path to your data file in the ptask folder
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptask", "data_3_25_40_66.dat")
    
    # Ensure the data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please check the path and ensure the file exists")
        # Try to list files in ptask directory
        ptask_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptask")
        if os.path.exists(ptask_dir):
            print(f"Files in {ptask_dir}:")
            for file in os.listdir(ptask_dir):
                print(f"  - {file}")
            # Try to use the first .dat file found
            dat_files = [f for f in os.listdir(ptask_dir) if f.endswith('.dat')]
            if dat_files:
                data_file = os.path.join(ptask_dir, dat_files[0])
                print(f"Using {dat_files[0]} instead")
        else:
            print(f"Directory {ptask_dir} not found")
            exit(1)
    
    # Run the MOGA
    moga = run_shift_scheduling_moga(data_file, generations=50)
    
    if moga is not None:
        # Get best solution for minimizing shifts
        best_solution = moga.archive.get_best_solution(objective_idx=0)
        
        if best_solution:
            print("\nBest solution for minimizing shifts:")
            print(f"- Number of shifts: {best_solution.fitness_values[0]}")
            print(f"- Jobs covered: {-best_solution.fitness_values[1]}")
            print(f"- Overlap penalty: {best_solution.fitness_values[2]}")
        
            # Save visualizations
            try:
                # Make sure output directory exists
                output_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Print current working directory
                print(f"Current working directory: {os.getcwd()}")
                print(f"Saving files to: {output_dir}")
                
                # Create and save each visualization with full path
                pareto_front_path = os.path.join(output_dir, "pareto_front.png")
                moga.visualize_pareto_front().savefig(pareto_front_path)
                print(f"Saved Pareto front visualization to {pareto_front_path}")
                
                pareto_3d_path = os.path.join(output_dir, "pareto_front_3d.png")
                moga.visualize_pareto_front_3d().savefig(pareto_3d_path)
                print(f"Saved 3D Pareto front visualization to {pareto_3d_path}")
                
                convergence_path = os.path.join(output_dir, "convergence.png")
                moga.visualize_convergence().savefig(convergence_path)
                print(f"Saved convergence visualization to {convergence_path}")
                
                solution_path = os.path.join(output_dir, "best_solution.png")
                moga.visualize_solution(best_solution).savefig(solution_path)
                print(f"Saved best solution visualization to {solution_path}")
                
                print("\nAll visualizations saved successfully.")
            except Exception as e:
                print(f"Error saving visualizations: {e}")
        else:
            print("\nNo valid solution found for minimizing shifts.")
    else:
        print("\nMOGA execution failed.")