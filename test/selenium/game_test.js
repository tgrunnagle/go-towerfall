const { Builder, By, Key, until } = require('selenium-webdriver');
const readline = require('readline');

async function waitForUserInput(message = 'Press Enter to exit...') {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    return new Promise(resolve => rl.question(message, ans => {
        rl.close();
        resolve(ans);
    }));
}

async function waitForElement(driver, xpath, timeout = 10000) {
    console.log(`Waiting for element: ${xpath}`);
    try {
        await driver.wait(until.elementLocated(By.xpath(xpath)), timeout);
        const element = await driver.findElement(By.xpath(xpath));
        console.log(`Found element: ${xpath}`);
        return element;
    } catch (error) {
        console.error(`Failed to find element: ${xpath}`);
        console.error(error);
        throw error;
    }
}

async function waitForElementText(driver, xpath, timeout = 10000) {
    console.log(`Waiting for element text: ${xpath}`);
    await waitForElement(driver, xpath, timeout);
    const element = await driver.findElement(By.xpath(xpath));
    const startTime = performance.now();
    while (performance.now() - startTime < timeout) {
        const text = await element.getText();
        if (text) {
            console.log(`Found element text: ${text}`);
            return text;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    console.error(`Failed to find element text: ${xpath}`);
    throw new Error(`Failed to find element text: ${xpath}`);
}

async function createGame(driver) {
    try {
        // Navigate to the game page
        console.log('Navigating to game page...');
        await driver.get('http://localhost:4001');

        // Wait for and click "Create New Game" button
        const createButton = await waitForElement(driver, "//button[contains(text(), 'Create New Game')]");
        await createButton.click();
        console.log('Clicked Create New Game button');

        // Wait for and fill in room name
        const roomNameInput = await waitForElement(driver, "//input[@placeholder='Enter room name']");
        await roomNameInput.sendKeys('Test Room');
        console.log('Entered room name');

        // Wait for and fill in player name
        const playerNameInput = await waitForElement(driver, "//input[@placeholder='Enter your name']");
        await playerNameInput.sendKeys('Trey');
        console.log('Entered player name');

        // Wait for and click "Start Game" button
        const startButton = await waitForElement(driver, "//button[contains(text(), 'Start Game')]");
        await startButton.click();
        console.log('Clicked Start Game button');

        // Wait for room code to appear
        await waitForElement(driver, "//p[contains(text(), 'Room Code:')]/strong");
        console.log('Room code appeared');

        // Extract room code and password
        const roomCode = await waitForElementText(driver, "//p[contains(text(), 'Room Code:')]/strong");
        const roomPassword = await waitForElementText(driver, "//p[contains(text(), 'Password:')]/strong");
        console.log('Extracted room code and password');

        return { roomCode, roomPassword };
    } catch (error) {
        console.error('Error in createGame:', error);
        throw error;
    }
}

async function joinGame(driver, roomCode, roomPassword) {
    try {
        // Navigate to the game page
        console.log('Navigating to game page...');
        await driver.get('http://localhost:4001');

        // Wait for and click "Join Existing Game" button
        const joinButton = await waitForElement(driver, "//button[contains(text(), 'Join Existing Game')]");
        await joinButton.click();
        console.log('Clicked Join Existing Game button');

        // Wait for and fill in room code
        const roomCodeInput = await waitForElement(driver, "//input[@placeholder='Enter room code']");
        await roomCodeInput.sendKeys(roomCode);
        console.log('Entered room code');

        // Wait for and fill in room password
        const roomPasswordInput = await waitForElement(driver, "//input[@placeholder='Enter room password']");
        await roomPasswordInput.sendKeys(roomPassword);
        console.log('Entered room password');

        // Wait for and fill in player name
        const playerNameInput = await waitForElement(driver, "//input[@placeholder='Enter your name']");
        await playerNameInput.sendKeys('Bob');
        console.log('Entered player name');

        // Wait for and click "Join Game" button
        const joinGameButton = await waitForElement(driver, "//button[contains(text(), 'Join Game')]");
        await joinGameButton.click();
        console.log('Clicked Join Game button');

        // Wait for room code to appear
        await waitForElementText(driver, "//p[contains(text(), 'Room Code:')]/strong");
        console.log('Room code appeared');
    } catch (error) {
        console.error('Error in joinGame:', error);
        throw error;
    }
}

async function runTest() {
    let driver1 = null;
    let driver2 = null;

    try {
        // Create two browser instances
        console.log('Creating browser instances...');
        driver1 = await new Builder().forBrowser('chrome').build();
        driver2 = await new Builder().forBrowser('chrome').build();

        // Create game in first browser
        console.log('Creating game...');
        const { roomCode, roomPassword } = await createGame(driver1);
        console.log(`Game created with Room Code: ${roomCode}, Password: ${roomPassword}`);

        // Join game in second browser
        console.log('Joining game...');
        await joinGame(driver2, roomCode, roomPassword);
        console.log('Game joined successfully');

        // Wait for user input before closing
        await waitForUserInput();
    } catch (error) {
        console.error('Test failed:', error);
        throw error;
    } finally {
        // Clean up
        console.log('Cleaning up...');
        if (driver1) await driver1.quit();
        if (driver2) await driver2.quit();
    }
}

// Run the test
runTest().catch(error => {
    console.error('Test failed with error:', error);
    process.exit(1);
});
