#!/bin/sh
# Script to test connectivity from the UI container to the backend

echo "Testing network connectivity to backend..."

# Test function to try connection and save results
test_connection() {
  local url=$1
  local desc=$2
  local output_file="/tmp/test_output_$RANDOM.txt"
  
  echo "\n🔍 Testing $desc: $url"
  if wget -q --timeout=5 --tries=1 -O "$output_file" "$url"; then
    echo "✅ SUCCESS: Connected to $url"
    echo "Response content (first 100 chars):"
    head -c 100 "$output_file"
    echo "\n"
    rm -f "$output_file"
    return 0
  else
    echo "❌ FAILED: Could not connect to $url"
    rm -f "$output_file"
    return 1
  fi
}

# Track if any connection succeeded
SUCCESS=0

# Try with service name (service discovery)
if test_connection "http://app:8000/health" "service name 'app'"; then
  SUCCESS=1
fi

# Try container name
if test_connection "http://reelgenius-backend:8000/health" "container name"; then
  SUCCESS=1
fi

# Try with host.docker.internal (for macOS/Windows Docker Desktop)
if test_connection "http://host.docker.internal:8000/health" "host.docker.internal"; then
  SUCCESS=1
fi

# Try localhost (works if port is mapped or in host networking mode)
if test_connection "http://localhost:8000/health" "localhost"; then
  SUCCESS=1
fi

# Try to verify our own health endpoints are working
echo "\n🔍 Testing our own health endpoints..."

# Try App Router health endpoint
if wget -q --timeout=2 --tries=1 -O "/tmp/app_router_health.txt" "http://localhost:3000/api/health"; then
  echo "✅ SUCCESS: App Router health endpoint is working correctly"
  cat "/tmp/app_router_health.txt" | head -c 100
  echo "\n"
  rm -f "/tmp/app_router_health.txt"
else
  echo "⚠️ WARNING: App Router health endpoint is not responding yet. This is expected during startup."
fi

# Try Pages Router health endpoint
if wget -q --timeout=2 --tries=1 -O "/tmp/pages_router_health.txt" "http://localhost:3000/api/healthcheck"; then
  echo "✅ SUCCESS: Pages Router health endpoint is working correctly"
  cat "/tmp/pages_router_health.txt" | head -c 100
  echo "\n"
  rm -f "/tmp/pages_router_health.txt"
else
  echo "⚠️ WARNING: Pages Router health endpoint is not responding yet. This is expected during startup."
fi

# Try the static file as fallback
if wget -q --timeout=2 --tries=1 -O "/dev/null" "http://localhost:3000/healthcheck.txt"; then
  echo "✅ SUCCESS: Static health check file is accessible"
else
  echo "⚠️ WARNING: Static health check file is not accessible yet"
fi

# Try container IP
echo "\n📌 Trying to get backend container IP:"
APP_IP=$(getent hosts app | awk '{ print $1 }')
echo "Resolved 'app' to IP: $APP_IP"
if [ -n "$APP_IP" ]; then
  if test_connection "http://$APP_IP:8000/health" "container IP"; then
    SUCCESS=1
  fi
fi

# Try to get gateway
echo "\n📋 Network interfaces:"
ip addr

echo "\n📋 Routing table:"
ip route

echo "\n📋 DNS configuration:"
cat /etc/resolv.conf

echo "\n📋 Hosts file:"
cat /etc/hosts

# Print success or failure message
if [ $SUCCESS -eq 1 ]; then
  echo "\n✅ CONNECTIVITY TEST RESULT: At least one method worked! The UI should be able to connect to the backend."
else
  echo "\n❌ CONNECTIVITY TEST RESULT: All methods failed. UI may not be able to connect to the backend."
  echo "Please check your Docker configuration and network settings."
fi

echo "\nConnectivity test completed."