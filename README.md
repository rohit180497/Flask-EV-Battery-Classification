# Flask-EV-Battery-Classification

The objective of this project is to address battery failures in electric vehicles using a physics-guided supervised learning approach for classification. The goal is to develop an advanced battery health management system that integrates machine learning techniques with domain-specific knowledge to accurately predict and classify battery health conditions.

### Dataset Overview:
Our dataset is made up of 10,000 rows and contains 55 different columns.

#### Data Generation Process:
To build this dataset, I had to generate synthetic data using Python. Considering research paper on EV Battery Health 
[https://ieeexplore.ieee.org/document/8882735]

I put in the effort to ensure that the synthetic data resembled real world-related data and its features.
Absence of Real-Life Patterns:
It's important to note that our synthetic dataset doesn't naturally exhibit the patterns or correlations we'd typically find in real-world data.
As a result, I don't anticipate finding strong linear relationships or causal connections within the dataset.

### Meta Data Description :

1. `Timestamp`: The timestamp when the data point was recorded.
2. `State_of_Charge`: Represents the current charge level of the battery, ranging from 0.2 (low charge) to 1.0 (full charge).
3. `State_of_Health`: Indicates the health status of the battery, with values ranging from 0.8 (poor health) to 1.0 (excellent health).
4. `Voltage`: The electrical voltage of the battery, typically between 370.0V and 430.0V.
5. `Current`: Represents the electric current flowing through the battery, ranging from 10.0A to 80.0A.
6. `Temperature`: The temperature of the battery, with values between -10.0°C (cold) and 100.0°C (hot).
7. `Cycling_Count`: The number of charge-discharge cycles the battery has undergone, ranging from 100 to 1500.
8. `Internal_Resistance`: Represents the internal resistance of the battery, which can affect its performance, with values between 0.5 and 5.0.
9. `Rate_of_Charge`: Indicates the rate at which the battery is being charged, ranging from 0.2 to 0.8.
10. `Charge_Discharge_Efficiency`: Represents the efficiency of the battery's charge and discharge processes, with values between 0.7 and 0.95.
11. `Depth_of_Discharge`: Indicates how deeply the battery is discharged during its use, ranging from 0.1 (shallow discharge) to 0.6 (deep discharge).
12. `Charge_Rate`: Represents the rate at which the battery is being charged, typically between 1.0 and 6.0.
13. `Discharge_Rate`: Indicates the rate at which the battery is being discharged, with values between 1.0 and 6.0.
14. `Voltage_Sag`: Represents the temporary voltage drop during battery operation, typically between 0.0V and 5.0V.
15. `Cell_Balancing_Status`: A binary value (0 or 1) indicating whether cell balancing is active or not.
16. `Self_Discharge_Rate`: Indicates the rate at which the battery self-discharges when not in use, typically between 0.001 and 0.01.
17. `EIS_Data`: Represents data related to Electrochemical Impedance Spectroscopy (EIS) measurements, with values between 10.0 and 80.0.
18. `Cycle_Life`: Indicates the expected number of charge-discharge cycles the battery can endure, ranging from 300 to 2000.
19. `Max_Charge_Power`: The maximum power at which the battery can be charged, typically between 20.0 and 60.0 Watts.
20. `Max_Discharge_Power`: The maximum power at which the battery can be discharged,
typically between 20.0 and 60.0 Watts.
21. `Anode_Degradation`: Represents the degradation level of the battery's anode, with values between 0.2 and 1.0.
22. `Cathode_Degradation`: Indicates the degradation level of the battery's cathode, ranging from 0.2 to 1.0.
23. `Electrolyte_Level`: Represents the level of electrolyte in the battery, typically between 30% and 90%.
24. `Chemical_Composition`: Indicates the chemical composition of the battery, with values ranging from 40% to 90%.
25. `Rate_of_Charge_Peak`: Indicates the peak rate of charge, typically between 0.5 and 1.0.
26. `Rate_of_Discharge_Peak`: Represents the peak rate of discharge, with values between 0.5 and 1.0.
27. `Calendar_Aging`: A binary value (0 or 1) indicating whether calendar aging is active or not.
28. `Cell_Voltage_Balance`: Represents the voltage balance among the battery cells, typically between 0.8 and 1.0.
29. `Altitude_Change`: Indicates changes in altitude during battery operation, with values between 0.8 and 1.0.
30. `Tire_Pressure`: Represents the tire pressure for a vehicle, typically between 25.0 and 35.0 psi.
31. `Regenerative_Braking`: Indicates the use of regenerative braking, with values between 0.2 and 0.8.
32. `Load_Weight`: Represents the weight of the load being carried, typically between 10.0 and 60.0 kg.
33. `Road_Inclination`: Indicates the road inclination (slope) in degrees, ranging from -5.0 to 5.0.
34. `Idle_Time`: Represents the time the vehicle/engine is idle, typically between 0 and 6 hours.
35. `Charging_Time`: Represents the time taken for charging the battery, typically between 1 and 12 hours.
36. `Charging_Frequency`: Indicates how often charging occurs, typically between 1 and 6 times.
37. `Payload_Distribution`: Represents the distribution of payload, typically between 0.4 (rear-heavy) and 0.6 (evenly distributed).
38. `Payload_Changes`: Indicates changes in payload, with values between 0.2 and 0.8.
39. `Trailer_Attachment`: A binary value (0 or 1) indicating whether a trailer is attached or not.
40. `Start_Stop_Frequency`: Indicates the frequency of engine start-stop events, typically between 1 and 5 times.
41. `Auxiliary_Power_Usage`: Represents the usage of auxiliary power, typically between 0.2 and 0.8.
42. `Truck_Speed`: The speed of the truck, typically between 20.0 and 100.0 km/h.
43. `Acceleration`: Represents the acceleration rate of the vehicle, typically between 0.2 and 3.0 m/s².
44. `Climate_Control_Usage`: Indicates the usage of climate control systems, typically between 0.2 and 0.8.
45. `Electrolyte_Type`: Categorical feature indicating the type of battery electrolyte (Lithium-ion, Lithium-polymer, Solid-State).
46. `Charging_Infrastructure`: Categorical feature indicating the type of charging infrastructure (Level 1, Level 2, DC Fast Charger).
47. `Road_Type`: Categorical feature indicating the type of road (Highway, City, Rural).
48. `Driving_Pattern`: Categorical feature indicating the driving pattern (Aggressive, Moderate, Gentle).
49. `Weather_Conditions`: Categorical feature indicating the weather conditions (Sunny, Rainy, Snowy).
50. `Battery_Chemistry`: Categorical feature indicating the battery chemistry (NMC, LFP, NCA, LCO).
51. `Truck_Model`: Categorical feature indicating the model of the truck (Model A, Model B, Model C).
52. `Trailer_Attached`: Categorical feature indicating whether a trailer is attached (Yes, No).
53. `Driver_Experience`: Categorical feature indicating the driver's experience level (Novice, Intermediate, Experienced).
54. Battery Health : Health of the battery in percentage which decides Battery Class (Target Variable)

