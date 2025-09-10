import re
from datetime import datetime
from aiopath import AsyncPath


async def find_sequential_files(self):
    # Use current date for the path
    current_date = datetime.now().strftime("%Y%m%d")
    path = AsyncPath(f"./uploaded_images")
    files_with_timestamps = []
    
    timestamp_pattern = r'(\d{8}_\d{6})'
    
    print("Scanning uploaded_images directory...")
    async for file in path.iterdir():
        if await file.is_file():
            filename = file.name
            print(f"Found file: {filename}")
            match = re.search(timestamp_pattern, filename)
            if match:
                timestamp_str = match.group(1)
                try:
                    dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    files_with_timestamps.append((file, dt, timestamp_str))
                    print(f"  Added to processing: {filename} ({timestamp_str})")
                except ValueError:
                    print(f"  Skipped (invalid timestamp): {filename}")
                    continue
            else:
                print(f"  Skipped (no timestamp pattern): {filename}")
    
    print(f"Total files with timestamps: {len(files_with_timestamps)}")
    files_with_timestamps.sort(key=lambda x: x[1])
    sequential_groups = []
    
    # Group consecutive files within time threshold
    used_indices = set()
    
    for i, (file1, dt1, ts1) in enumerate(files_with_timestamps):
        if i in used_indices:
            continue
            
        group = [file1]
        used_indices.add(i)
        
        for j in range(i + 1, len(files_with_timestamps)):
            if j in used_indices:
                continue
                
            file2, dt2, ts2 = files_with_timestamps[j]
            time_diff = abs((dt2 - dt1).total_seconds())
            
            if time_diff <= 40:  # Within 10 seconds of first file
                group.append(file2)
                used_indices.add(j)
            elif time_diff > 40:  
                break
        
        if len(group) >= 1:
            print(f"Created group with {len(group)} files: {[f.name for f in group]}")
            sequential_groups.append(group)
    
    print(f"Total groups created: {len(sequential_groups)}")
    return sequential_groups