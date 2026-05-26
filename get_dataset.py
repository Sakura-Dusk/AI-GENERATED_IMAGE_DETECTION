import kagglehub

# Download latest version
path = kagglehub.dataset_download("cartografia/unbiased-tiny-genimage")

print("Path to dataset files:", path)