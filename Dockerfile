FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome (latest)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver (matching Chrome 146)
RUN wget -q https://storage.googleapis.com/chrome-for-testing-public/146.0.7680.165/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf chromedriver-linux64*

# Create selenium cache directory with proper permissions
RUN mkdir -p /.cache/selenium && chmod 777 /.cache/selenium

# Set environment variables
ENV SELENIUM_REMOTE_URL=http://selenium-hub:4444/wd/hub
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 10000

CMD ["python", "main.py"]
