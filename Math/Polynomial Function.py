import matplotlib.pyplot as plt

# Data
years = [2023, 2022, 2021, 2020, 2019,2018,2017,2016,2015,2014]
revenue = [490675, 433300, 331026, 308450, 374377,347684,313849,185943,288226,253667]

# Create the plot
plt.figure(figsize=(10, 10))
plt.plot(years, revenue, marker='o')

# Add titles and labels
plt.title("Revenue Growth from 2019 to 2023 ")
plt.xlabel("Year")
plt.ylabel("Revenue (RM'000)")

# Set x-axis ticks
plt.xticks(years)
    
# Show the plot
plt.grid(True)
plt.show()
