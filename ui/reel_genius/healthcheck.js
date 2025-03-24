#!/usr/bin/env node

/**
 * Health check script for Docker/Kubernetes
 * This script will check if the UI server is healthy by making a request to it.
 * 
 * It first tries the Pages API route at /api/health, and if that fails, 
 * it falls back to the static file at /healthcheck.txt.
 */

const http = require('http');

// Configuration
const HOST = process.env.HEALTH_CHECK_HOST || 'localhost';
const PORT = process.env.HEALTH_CHECK_PORT || 3000;
// Use the appropriate health check paths
const APP_ROUTER_PATH = '/api/health';           // App Router path
const PAGES_ROUTER_PATH = '/api/healthcheck';    // Pages Router path
const STATIC_FILE_PATH = '/healthcheck.txt';     // Static file fallback
const TIMEOUT = parseInt(process.env.HEALTH_CHECK_TIMEOUT) || 5000;

/**
 * Make a health check request
 * @param {string} path - The path to check
 * @param {boolean} isFallback - Whether this is a fallback request
 * @returns {Promise<boolean>} - Whether the check was successful
 */
function makeRequest(path, isFallback = false) {
  return new Promise((resolve) => {
    console.log(`Health check request to ${path}${isFallback ? ' (fallback)' : ''}`);
    
    // Create the request options
    const options = {
      host: HOST,
      port: PORT,
      path: path,
      timeout: TIMEOUT
    };

    // Make the request
    const req = http.request(options, (res) => {
      console.log(`Health check status from ${path}: ${res.statusCode}`);
      
      // Process the response
      if (res.statusCode === 200) {
        // The server is healthy
        resolve(true);
      } else {
        // The server is not healthy
        console.error(`Health check to ${path} failed: Received status code ${res.statusCode}`);
        resolve(false);
      }
    });

    // Handle timeout
    req.on('timeout', () => {
      console.error(`Health check to ${path} failed: Request timed out`);
      req.destroy();
      resolve(false);
    });

    // Handle errors
    req.on('error', (err) => {
      console.error(`Health check to ${path} failed: ${err.message}`);
      resolve(false);
    });

    // Send the request
    req.end();
  });
}

// Main execution
async function runHealthChecks() {
  try {
    // Try the App Router endpoint first
    const appRouterResult = await makeRequest(APP_ROUTER_PATH);
    
    if (appRouterResult) {
      console.log('App Router health check succeeded');
      process.exit(0);
    }
    
    // If the App Router check fails, try the Pages Router endpoint
    console.log('App Router health check failed, trying Pages Router...');
    const pagesRouterResult = await makeRequest(PAGES_ROUTER_PATH);
    
    if (pagesRouterResult) {
      console.log('Pages Router health check succeeded');
      process.exit(0);
    }
    
    // If both API endpoints fail, try the static file fallback
    console.log('API health checks failed, trying static file fallback...');
    const staticFileResult = await makeRequest(STATIC_FILE_PATH, true);
    
    if (staticFileResult) {
      console.log('Static file health check succeeded');
      process.exit(0);
    }
    
    // All checks failed
    console.error('All health checks failed');
    process.exit(1);
  } catch (error) {
    console.error('Unexpected error in health check:', error);
    process.exit(1);
  }
}

// Run the health checks
runHealthChecks();