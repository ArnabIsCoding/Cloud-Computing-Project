import os
import pandas as pd
import matplotlib.pyplot as plt

def create_visualizations():
    print("Booting up Visualization Engine...")
    analytics_dir = "data/analytics"
    visuals_dir = f"{analytics_dir}/visuals"

    # Create the folder for our images
    os.makedirs(visuals_dir, exist_ok=True)

    # --- Chart 1: Busiest Airports ---
    busiest_path = f"{analytics_dir}/q1_1_busiest_airports.parquet"
    if os.path.exists(busiest_path):
        print("Generating 'Busiest Airports' Chart...")
        df_busiest = pd.read_parquet(busiest_path)

        plt.figure(figsize=(10, 6))
        # Draw the bar chart
        bars = plt.bar(df_busiest['airport'], df_busiest['total_flights'], color='#1f77b4')
        plt.title('Top 5 Busiest Airports (2008)', fontsize=16, fontweight='bold')
        plt.xlabel('Airport Code', fontsize=12)
        plt.ylabel('Total Flights (Departures + Arrivals)', fontsize=12)

        # Add the exact numbers on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2000, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{visuals_dir}/1_busiest_airports.png", dpi=300)
        plt.close()
        print("Saved: 1_busiest_airports.png")

    # --- Chart 2: Best Airlines at SFO ---
    airlines_path = f"{analytics_dir}/q2_1_top_airlines_by_airport.parquet"
    if os.path.exists(airlines_path):
        print("Generating 'Best Airlines at SFO' Chart...")
        df_airlines = pd.read_parquet(airlines_path)

        # Filter just for SFO and grab the top 5
        sfo_only = df_airlines[df_airlines['Origin'] == 'SFO'].head(5)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(sfo_only['UniqueCarrier'], sfo_only['avg_delay'], color='#2ca02c')
        plt.title('Most Punctual Airlines departing SFO', fontsize=16, fontweight='bold')
        plt.xlabel('Airline Code', fontsize=12)
        plt.ylabel('Average Departure Delay (Minutes)', fontsize=12)

        # Add the exact numbers on top
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval} min", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{visuals_dir}/2_sfo_best_airlines.png", dpi=300)
        plt.close()
        print("Saved: 2_sfo_best_airlines.png")

    print(f"\n All dashboards generated! Open '{visuals_dir}' in VS Code to view them.")

if __name__ == "__main__":
    create_visualizations()
