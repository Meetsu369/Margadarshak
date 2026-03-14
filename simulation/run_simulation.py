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

            # Congestion detection
            if avg_speed < 5 and len(vehicles) > 0:
                print("⚠ Congestion detected!")

            speed_history.append(avg_speed)
            vehicle_history.append(len(vehicles))

            writer.writerow([step, len(vehicles), avg_speed])

            print(f"Step {step} | Vehicles: {len(vehicles)} | Avg Speed: {avg_speed:.2f}")

            step += 1

    traci.close()

    plt.plot(speed_history)
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Speed")
    plt.title("Traffic Speed Over Time")
    plt.show()

if __name__ == "__main__":
    run()