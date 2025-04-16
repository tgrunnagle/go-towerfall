// Run during build to generate a config.js file in public that can be read by static js scripts
const fs = require('fs');
const path = require('path');

// Read .env file if it exists
require('dotenv').config();

const config = {
    BACKEND_WS_URL: process.env.BACKEND_WS_URL,
    BACKEND_API_URL: process.env.BACKEND_API_URL
};

const configScript = `window.APP_CONFIG = ${JSON.stringify(config, null, 2)};`;

// Write to public directory
fs.writeFileSync(
    path.join(__dirname, '../public/config.js'),
    configScript
);

console.log('Config file generated successfully!');
