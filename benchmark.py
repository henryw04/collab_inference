import psutil
import math


def calculate_max_layers():
    # 1. Get system RAM and convert to GB
    stats = psutil.virtual_memory()
    available_ram_gb = stats.available / (1024**3)

    # 2. Determine available RAM for inference (minus 1GB safety buffer)
    inference_ram_gb = available_ram_gb - 1.0

    if inference_ram_gb <= 0:
        print("Error: Not enough available RAM to allocate a 1GB buffer.")
        return

    print(f"System Available RAM: {available_ram_gb:.2f} GB")
    print(f"Target RAM for Inference: {inference_ram_gb:.2f} GB")
    print("-" * 30)

    # 3. Prompt for model size
    try:
        user_input = float(
            input("Enter model size in Billions (e.g., 3 for 3B, 0.5 for 0.5B): ")
        )
        total_params = user_input * 1_000_000_000
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    try:
        user_input = float(
            input("Enter model precision (e.g., 16 for FP16, 32 for FP32): ")
        )
        data_size = user_input / 8  # Convert bits to bytes
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    total_layers = 32 if user_input >= 7 else 26

    total_model_weight_gb = (total_params * data_size) / (1024**3)

    overhead_gb = total_model_weight_gb * 0.10
    layer_weight_gb = (total_model_weight_gb - overhead_gb) / total_layers

    max_layers = math.floor((inference_ram_gb - overhead_gb) / layer_weight_gb)

    max_layers = min(max_layers, total_layers)
    max_layers = max(max_layers, 0)  # Cannot be negative

    print("-" * 30)
    print(f"Model: {user_input}B Parameters")
    print(f"Estimated Total Weight (FP16): {total_model_weight_gb:.2f} GB")
    print(f"Estimated weight per layer: {layer_weight_gb * 1024:.2f} MB")
    print(f"\n>>> Recommended Max Layers for this device: {max_layers}")


if __name__ == "__main__":
    calculate_max_layers()
