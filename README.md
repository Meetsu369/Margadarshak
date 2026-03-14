\# Margadarshak



Margadarshak is an AI-driven intelligent traffic guidance and control system.



The system combines traffic simulation, graph-based learning, and reinforcement learning to optimize urban traffic flow.



\## Architecture Overview



Margadarshak is divided into four main layers:



1\. \*\*Sense (Input Layer)\*\*

&#x20;  - Traffic data is collected from the SUMO simulator.

&#x20;  - State vectors describing traffic conditions are generated.



2\. \*\*Think-A (Relational Intelligence)\*\*

&#x20;  - Road networks are represented as graphs.

&#x20;  - A Graph Neural Network (GNN) analyzes traffic relationships.



3\. \*\*Think-B (Decision Layer)\*\*

&#x20;  - A Deep Reinforcement Learning agent learns optimal signal policies.

&#x20;  - Rewards are based on congestion reduction.



4\. \*\*Act (Execution Layer)\*\*

&#x20;  - The learned policy controls traffic lights using the TraCI interface.



\## Project Structure



simulation/      SUMO traffic simulation files  

sensing/         Traffic state extraction modules  

intelligence/    Graph Neural Network models  

decision/        Reinforcement learning agents  

training/        Model training pipelines  

dashboard/       Monitoring and visualization  

experiments/     Experimental prototypes  

docs/            System documentation  



\## Goal



To build a real-world capable intelligent traffic management system capable of reducing congestion and improving traffic efficiency.



