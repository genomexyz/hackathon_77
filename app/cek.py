def determine_quadrant(value, percentiles):
    # Ensure percentiles are sorted
    percentiles = sorted(percentiles)
    
    for i, p in enumerate(percentiles):
        if value < p:
            return i
    return len(percentiles)

# Example usage:
percentiles = [5, 50, 95]  # This can be any number of percentiles
value = 4
quadrant = determine_quadrant(value, percentiles)
print(f"The value {value} is in quadrant {quadrant}.")
