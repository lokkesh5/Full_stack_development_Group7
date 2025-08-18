# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install deps with cache-friendly layers
COPY package*.json /app/
RUN npm ci

# Copy app source
COPY . /app

# Vite dev server port
EXPOSE 5173

# Run dev server; compose will set VITE_API_BASE in environment
CMD ["npm", "run", "dev", "--", "--host"]
