import kagglehub

# Download latest version
path = kagglehub.dataset_download("luluw8071/brain-tumor-mri-datasets")

print("Path to dataset files:", path)