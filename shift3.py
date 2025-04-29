import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import pandas as pd
import re

# Clear any existing creator instances to avoid duplication errors
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

# Define the problem as a multi-objective problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Functions for parsing the data file
def parse_data_file(file_path):
    """
    Parse the data file with the format specified in the problem.
    """
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Extract multi-skilling level
        multi_skilling_match = re.search(r"Multi-skilling level = (\d+)", lines[0])
        multi_skilling_level = int(multi_skilling_match.group(1)) if multi_skilling_match else None
        
        # Extract type
        type_match = re.search(r"Type = (\d+)", lines[2])
        problem_type = int(type_match.group(1)) if type_match else None
        
        # Extract number of jobs
        jobs_match = re.search(r"Jobs = (\d+)", lines[3])
        jobs = int(jobs_match.group(1)) if jobs_match else None
        
        # Extract start and end times
        start_end_times = []
        for i in range(4, 4 + jobs):
            if i < len(lines):
                values = list(map(int, lines[i].strip().split()))
                if len(values) >= 2:  # Ensure we have at least start and end time
                    start_end_times.append(values)
        
        # Extract qualifications
        qualifications_line_idx = 4 + jobs
        qualifications_match = re.search(r"Qualifications = (\d+)", lines[qualifications_line_idx])
        qualifications = int(qualifications_match.group(1)) if qualifications_match else None
        
        # Extract qualified jobs
        qualified_jobs = []
        for i in range(qualifications_line_idx + 1, qualifications_line_idx + 1 + qualifications):
            if i < len(lines):
                match = re.search(r"^\s*\d+:((?:\s+\d+)+)", lines[i])
                if match:
                    jobs_list = list(map(int, match.group(1).strip().split()))
                    qualified_jobs.append(jobs_list)
        
        # Calculate due dates if provided in start_end_times
        due_dates = []
        for job_info in start_end_times:
            if len(job_info) >= 3:  # If there's a third value, it's the due date
                due_dates.append(job_info[2])
            else:
                # Default due date is end time if not specified
                due_dates.append(job_info[1])
        
        # Calculate processing times
        processing_times = [end - start for start, end in [(job[0], job[1]) for job in start_end_times]]
        
        return {
            'multi_skilling_level': multi_skilling_level,
            'type': problem_type,
            'jobs': jobs,
            'qualifications': qualifications,
            'start_end_times': start_end_times,
            'qualified_jobs': qualified_jobs,
            'due_dates': due_dates,
            'processing_times': processing_times
        }
    except Exception as e:
        print(f"Error parsing data file: {e}")
        return None

def calculate_shifts(data):
    """
    Calculate the number of possible shifts based on the problem data.
    """
    # The number of shifts can be estimated in various ways:
    # 1. As the maximum possible end time in the schedule
    max_end_time = max(job[1] for job in data['start_end_times'])
    
    # 2. As the number of qualifications (if each qualification corresponds to a shift)
    num_qualifications = data['qualifications']
    
    # Choose the appropriate method based on the problem
    # Here we'll use a combination of both approaches
    return max(max_end_time, num_qualifications * 2)

# Evaluation functions for the multi-objective problem
def num_shifts(individual):
    """Count the number of unique shifts used in the schedule."""
    return len(set(individual))

def create_job_qualification_matrix(data):
    """Create a matrix of jobs x qualifications."""
    jobs = data['jobs']
    qualifications = data['qualifications']
    
    # Initialize matrix with zeros
    matrix = np.zeros((jobs, qualifications))
    
    # Fill the matrix
    for q, qual_jobs in enumerate(data['qualified_jobs']):
        for j in qual_jobs:
            if 0 <= j < jobs:  # Ensure job index is valid
                matrix[j][q] = 1
    
    return matrix

def create_shift_qualification_matrix(data, n_shifts):
    """Create a matrix of shifts x qualifications."""
    qualifications = data['qualifications']
    
    # Initialize matrix with zeros
    matrix = np.zeros((n_shifts, qualifications))
    
    # For simplicity, we'll assume each shift has one qualification
    # This can be modified based on the actual problem requirements
    for s in range(n_shifts):
        q = s % qualifications  # Assign qualifications in a round-robin fashion
        matrix[s][q] = 1
    
    # Apply multi-skilling: each shift can have multiple qualifications
    # based on the multi-skilling level
    multi_skilling_level = data['multi_skilling_level']
    if multi_skilling_level > 1:
        for s in range(n_shifts):
            for _ in range(1, multi_skilling_level):
                # Add additional qualifications to this shift
                available_quals = [q for q in range(qualifications) if matrix[s][q] == 0]
                if available_quals:
                    additional_qual = np.random.choice(available_quals)
                    matrix[s][additional_qual] = 1
    
    return matrix

def assignment_cost(individual, job_qualification_matrix, shift_qualification_matrix):
    """Calculate the cost of assigning jobs to shifts based on qualifications."""
    total_cost = 0
    
    for job_idx, shift in enumerate(individual):
        # Check if the shift has the necessary qualifications for this job
        required_quals = job_qualification_matrix[job_idx]
        shift_quals = shift_qualification_matrix[shift]
        
        # Calculate the mismatch cost (how many required qualifications are missing)
        mismatches = np.sum(np.logical_and(required_quals == 1, shift_quals == 0))
        total_cost += mismatches
    
    return total_cost

def calculate_completion_times(individual, start_end_times):
    """Calculate the completion time for each job based on the shift assignments."""
    n_jobs = len(individual)
    processing_times = [end - start for start, end in [(job[0], job[1]) for job in start_end_times]]
    
    # Initialize completion times for each shift
    shift_completion_times = {}
    
    # Schedule jobs on their assigned shifts
    completion_times = []
    
    for job_idx in range(n_jobs):
        shift = individual[job_idx]
        proc_time = processing_times[job_idx]
        earliest_start = start_end_times[job_idx][0]
        
        # Get the current completion time for this shift
        current_time = shift_completion_times.get(shift, 0)
        
        # Job can't start before its earliest start time
        start_time = max(current_time, earliest_start)
        
        # Calculate completion time
        completion_time = start_time + proc_time
        
        # Update the shift's completion time
        shift_completion_times[shift] = completion_time
        
        # Store the job's completion time
        completion_times.append(completion_time)
    
    return completion_times

def tardiness(individual, start_end_times, due_dates):
    """Calculate the total tardiness of jobs based on their completion times."""
    total_tardiness = 0
    
    # For each job, calculate its completion time based on the assigned shift
    completion_times = calculate_completion_times(individual, start_end_times)
    
    # Calculate tardiness for each job
    for job_idx, completion_time in enumerate(completion_times):
        job_tardiness = max(0, completion_time - due_dates[job_idx])
        total_tardiness += job_tardiness
    
    return total_tardiness

# Main genetic algorithm implementation
def setup_toolbox(data, n_shifts):
    """Set up the DEAP toolbox for the problem."""
    toolbox = base.Toolbox()
    
    # Create individuals with random integers between 0 and number_of_shifts-1
    toolbox.register("attr_int", random.randint, 0, n_shifts-1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_int, n=data['jobs'])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Create shift qualification matrix
    shift_qualification_matrix = create_shift_qualification_matrix(data, n_shifts)
    
    # Register the evaluation function
    def evaluate_individual(individual):
        job_qualification_matrix = create_job_qualification_matrix(data)
        shifts = num_shifts(individual)
        cost = assignment_cost(individual, job_qualification_matrix, shift_qualification_matrix)
        tard = tardiness(individual, data['start_end_times'], data['due_dates'])
        return (shifts, cost, tard)
    
    toolbox.register("evaluate", evaluate_individual)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_shifts-1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox

def run_ga(toolbox, n_gen=100, pop_size=100, cx_prob=0.5, mut_prob=0.2):
    """Run the genetic algorithm."""
    # Initialize population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Store statistics for each generation
    stats = []
    
    # Run the genetic algorithm
    for gen in range(n_gen):
        # Select offspring
        offspring = toolbox.select(population)