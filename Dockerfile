# Use an official Jupyter base image with Python and Jupyter installed
FROM jupyter/base-notebook:latest

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Switch to root user to install system packages
USER root

# Install build-essential, git, and libomp-dev
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libomp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to the notebook user
USER $NB_UID

# Install the Python package from GitHub
RUN pip install --no-cache-dir git+https://github.com/RodrigoZepeda/ABM_AMRO

# Expose the port Jupyter uses
EXPOSE 8888

# Set the command to start Jupyter
CMD ["start-notebook.sh", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
