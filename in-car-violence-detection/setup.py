from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="in-car-violence-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time violence and weapon detection for vehicle cabins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/depi_ai_final_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch==2.2.1",
        "torchvision==0.17.1",
        "ultralytics==8.1.0",
        "insightface==0.7.3",
        "onnxruntime==1.16.3",
        "opencv-python==4.8.1.78",
        "pillow==10.2.0",
        "numpy==1.24.4",
        "scipy==1.11.3",
        "albumentations==1.3.1",
        "tqdm==4.66.1",
        "pyyaml==6.0.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "jupyterlab>=4.0.0",
        "pytest>=7.4.0",
    ],
)
