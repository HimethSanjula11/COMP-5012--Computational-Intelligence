import pandas as pd
import re

# Load the data from the file
file_path = "ptask\\data_3_25_40_66.dat"
data = pd.read_csv(file_path)  # Adjust delimiter if necessary

# Display the first few rows of the data
print(data)



# Extract the multi-skilling level from the data

# Assuming the multi-skilling level is mentioned in the first row of the data
multi_skilling_level_match = re.search(r"Multi-skilling level = (\d+)", data.iloc[0, 0])
if multi_skilling_level_match:
    multi_skilling_level = int(multi_skilling_level_match.group(1))
    print("Extracted Multi-skilling level:", multi_skilling_level)
else:
    print("Multi-skilling level not found.")

    # Extract the Type value from the data
type_match = re.search(r"Type = (\d+)", data.iloc[2, 0])
if type_match:
    extracted_type = int(type_match.group(1))
    print("Extracted Type:", extracted_type)
else:
    print("Type not found.")

# Extract the Jobs value from the data
jobs_match = re.search(r"Jobs = (\d+)", data.iloc[3, 0])
if jobs_match:
    jobs = int(jobs_match.group(1))
    print("Extracted Jobs:", jobs)
else:
    print("Jobs not found.")
# Extract the Qualifications value from the data
qualifications_match = re.search(r"Qualifications = (\d+)", data.iloc[jobs + 4, 0])
if qualifications_match:
    qualifications = int(qualifications_match.group(1))
    print("Extracted Qualifications:", qualifications)
else:
    print("Qualifications not found.")
# Extract the start and end times from the data
start_end_data = data.iloc[4:44, 0]  # Rows 4 to 43 contain the start and end times
start_end_list = [list(map(int, row.split())) for row in start_end_data]

print("Extracted Start and End Times:")
print(start_end_list)
# Extract lines containing qualified jobs
qualified_jobs_lines = data.iloc[jobs + 5:, 0]  # Adjust the starting row based on the data structure
qualified_jobs_pattern = re.compile(r"^\s*\d+:((?:\s+\d+)+)")

qualified_jobs = []
for line in qualified_jobs_lines:
    match = qualified_jobs_pattern.match(line)
    if match:
        jobs_list = list(map(int, match.group(1).split()))
        qualified_jobs.append(jobs_list)

print("Extracted Qualified Jobs:")
print(qualified_jobs)