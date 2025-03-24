FROM node:20-alpine

WORKDIR /app

# Install network troubleshooting tools
RUN apk add --no-cache wget curl iputils bind-tools

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Make our connectivity test script executable
RUN chmod +x ./docker-test-connectivity.sh

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Make healthcheck script executable
RUN chmod +x ./healthcheck.js

# Define healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD node /app/healthcheck.js || exit 1

# Start the application
CMD ["npm", "start"]