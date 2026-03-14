import traci
import matplotlib.pyplot as plt
import csv

# Command to start SUMO simulation
SUMO_CMD = ["sumo", "-c", "config.sumocfg"]

def run():

    # Start SUMO
    traci.start(SUMO_CMD)

    step = 0
    speed_history = []
    vehicle_history = []

    # Open CSV file to store traffic data
    with open("traffic_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "vehicles", "avg_speed"])

        # Run simulation loop
        while step < 100:

            traci.simulationStep()

            vehicles = traci.vehicle.getIDList()
            speeds = []

            # Collect speed of every vehicle
            for vid in vehicles:
                speed = traci.vehicle.getSpeed(vid)
                speeds.append(speed)

            # Compute average speed
            if len(speeds) > 0:
                avg_speed = sum(speeds) / len(speeds)
            else:
                avg_speed = 0

            # Detect congestion
            if avg_speed < 5 and len(vehicles) > 0:
                print("⚠ Congestion detected!")

                # Get traffic light IDs
                tls_ids = traci.trafficlight.getIDList()

                # Change traffic light phase
                for tls in tls_ids:
                    traci.trafficlight.setPhase(tls, 0)

            # Store history
            speed_history.append(avg_speed)
            vehicle_history.append(len(vehicles))

            # Write to CSV dataset
            writer.writerow([step, len(vehicles), avg_speed])

            # Print live simulation status
            print(f"Step {step} | Vehicles: {len(vehicles)} | Avg Speed: {avg_speed:.2f}")

            step += 1

    # Close SUMO
    traci.close()

    # Plot traffic speed graph
    plt.plot(speed_history)
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Speed")
    plt.title("Traffic Speed Over Time")
    plt.show()


if __name__ == "__main__":
    run()