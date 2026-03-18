import sounddevice as sd

print("=== All Audio Devices ===\n")
print(sd.query_devices())

print(f"\nDefault input:  index={sd.default.device[0]}")
print(f"Default output: index={sd.default.device[1]}")

d = sd.query_devices(sd.default.device[0])
print(f"\nDefault input device: {d['name']}")
print(f"  Max input channels: {d['max_input_channels']}")
print(f"  Default samplerate: {d['default_samplerate']}")
