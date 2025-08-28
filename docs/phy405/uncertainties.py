
# Multimeter
def DCV_multimeter(reading, v_range):
    return 0.0015/100 * reading + 0.0004/100 * v_range

# Multimeter
def ACV_multimeter(reading, v_range):
    return 0.0015/100 * reading + 0.0004/100 * v_range

r1_uncertainty = DCV_multimeter(26.5748, 100)
r2_uncertainty = DCV_multimeter(12.2843, 100)
r3_uncertainty = DCV_multimeter(21.3271, 100)
r4_uncertainty = DCV_multimeter(21.8600, 100)

uncertainties = [r1_uncertainty, r2_uncertainty, r3_uncertainty, r4_uncertainty]

for i in range(len(uncertainties)):
    print(f"R{i + 1}: delta_V = {uncertainties[i]:.1}")
