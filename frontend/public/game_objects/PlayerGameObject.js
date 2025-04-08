import { GameObjectWithPosition } from "./GameObject.js";
import { Constants } from "../Constants.js";

export class PlayerGameObject extends GameObjectWithPosition {
    constructor(serverState, clientState) {
        super(serverState, clientState);
        this.getPlayerColor = this.getPlayerColor.bind(this);
        this.render = this.render.bind(this);
        this.dead = false;
    }

    // Utility functions
    getPlayerColor() {
        const id = this.serverState['id']
        if (!id) return '#3498db'; // Default blue

        // Simple hash function to generate a color from player ID
        let hash = 0;
        for (let i = 0; i < id.length; i++) {
            hash = id.charCodeAt(i) + ((hash << 5) - hash);
        }

        // Convert to hex color
        let color = '#';
        for (let i = 0; i < 3; i++) {
            const value = (hash >> (i * 8)) & 0xFF;
            color += ('00' + value.toString(16)).slice(-2);
        }

        return color;
    }

    render(canvasCtx, timestamp) {
        if (this.serverState.dead) {
            return; // Don't render dead players
        }

        this.renderInternal(canvasCtx, timestamp);
    }

    renderInternal(canvasCtx, timestamp) {
        // Interpolate object position
        const { predictedX, predictedY } = this.interpPosition(timestamp);

        const playerRadius = this.serverState.rad || 20.0;

        // Draw a player as a circle
        canvasCtx.beginPath();
        canvasCtx.arc(predictedX, predictedY, playerRadius, 0, Math.PI * 2);
        canvasCtx.fillStyle = this.getPlayerColor();
        canvasCtx.fill();

        // Draw player name
        canvasCtx.fillStyle = '#000';
        canvasCtx.font = '12px Arial';
        canvasCtx.textAlign = 'center';
        canvasCtx.fillText(this.serverState.name, predictedX, predictedY - playerRadius - 5);

        // Draw velocity indicator
        if (this.serverState.dx !== 0 || this.serverState.dy !== 0) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(predictedX, predictedY);
            canvasCtx.lineTo(
                predictedX + this.serverState.dx / 10,
                predictedY + this.serverState.dy / 10,
            ); // Scale down for better visualization
            canvasCtx.strokeStyle = '#FF0000';
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
        }

        // Draw direction indicator
        const dir = this.interpDirection();
        if (dir != null) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(predictedX, predictedY);
            canvasCtx.lineTo(
                predictedX + Math.cos(dir) * (playerRadius + 4),
                predictedY + Math.sin(dir) * (playerRadius + 4)
            );
            canvasCtx.strokeStyle = '#0000FF';
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
        }

        // Highlight current player
        if (this.clientState.isCurrentPlayer) {
            canvasCtx.beginPath();
            canvasCtx.arc(predictedX, predictedY, playerRadius + 4, 0, Math.PI * 2);
            canvasCtx.strokeStyle = '#00FF00';
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
        }

        this.clientState.rendersSinceUpdate++;
    }

    onDiedAnimation(eventData) {

        const startTime = performance.now();
        let visible = true;
        let lastToggle = startTime;

        const blinkAnimation = (canvasCtx, timestamp) => {
            if ((timestamp - startTime) / 1000.0 >= Constants.PLAYER_DIED_ANIMATION_TIME_SEC) {
                return false; // Animation complete
            }

            // Toggle visibility based on blink rate
            if ((timestamp - lastToggle) / 1000.0 >= Constants.PLAYER_DIED_ANIMATION_BLINK_RATE_SEC) {
                visible = !visible;
                lastToggle = timestamp;
            }

            if (visible) {
                this.renderInternal(canvasCtx, timestamp);
            }

            return true; // Animation still active
        };

        return blinkAnimation;
    }
}
