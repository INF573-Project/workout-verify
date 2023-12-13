
def simple_moving_average(data, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        moving_averages = []
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            average = sum(window) / window_size
            moving_averages.append(average)

        return moving_averages
