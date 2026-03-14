import traci
import matplotlib.pyplot as plt
import csv

SUMO_CMD = ["sumo", "-c", "config.sumocfg"]

def run():
    traci.start(SUMO_CMD)

    step = 0
    speed_history = []
    vehicle_history = []

    with open("traffic_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "vehicles", "avg_speed"])

        while step < 100:
            traci.simulationStep()

            vehicles = traci.vehicle.getIDList()
            speeds = []

            for vid in vehicles:
                speed = traci.vehicle.getSpeed(vid)
                speeds.append(speed)

            if len(speeds) > 0:
                avg_speed = sum(speeds) / len(speeds)
            else:
                avg_speed = 0

            # quick congestion check
            if avg_speed < 5 and len(vehicles) > 0:
                print("⚠ Congestion detected!")

                tls_ids = traci.trafficlight.getIDList()
                for tls in tls_ids:
                    traci.trafficlight.setPhase(tls, 0)

            # store speed history
            speed_history.append(avg_speed)

            # keep only last 5 speeds
            if len(speed_history) > 5:
                speed_history.pop(0)

            smoothed_speed = sum(speed_history) / len(speed_history)

            # persistent congestion detection
            if smoothed_speed < 5 and len(vehicles) > 0:
                print("⚠ Persistent congestion detected!")

            vehicle_history.append(len(vehicles))

            writer.writerow([step, len(vehicles), avg_speed])

            print(f"Step {step} | Vehicles: {len(vehicles)} | Avg Speed: {avg_speed:.2f}")

            step += 1

    traci.close()

    plt.plot(vehicle_history)
    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Vehicles")
    plt.title("Vehicle Count Over Time")
    plt.show()


if __name__ == "__main__":
    run()