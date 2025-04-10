import { GameObjectWithPosition } from './GameObject.js';
import { Constants } from '../Constants.js';

export class ArrowGameObject extends GameObjectWithPosition {
    constructor(serverState, clientState) {
        super(serverState, clientState);
        this.render = this.render.bind(this);
    }

    render(canvasCtx, timestamp) {
        // Don't render if destroyed
        if (this.serverState.destroyed) {
            return;
        }

        // Interpolate position and direction based on server state
        const { predictedX, predictedY } = this.interpPosition(timestamp);
        const dir = this.interpDirection();

        // Check if arrow is grounded
        const isGrounded = this.getStatePreferClient('ag');

        if (isGrounded) {
            // Draw an 'X' for grounded arrows
            const size = 6;
            canvasCtx.beginPath();
            canvasCtx.strokeStyle = '#000000';
            canvasCtx.lineWidth = 2;
            
            // Draw first line of X
            canvasCtx.moveTo(predictedX - size, predictedY - size);
            canvasCtx.lineTo(predictedX + size, predictedY + size);
            
            // Draw second line of X
            canvasCtx.moveTo(predictedX + size, predictedY - size);
            canvasCtx.lineTo(predictedX - size, predictedY + size);
            
            canvasCtx.stroke();
        } else {
            // Draw arrow as triangle pointed in direction of travel
            const arrowLength = 12;
            const arrowWidth = 4;

            canvasCtx.save();
            canvasCtx.translate(predictedX, predictedY);
            canvasCtx.rotate(dir);

            canvasCtx.beginPath();
            canvasCtx.moveTo(arrowLength, 0);  // tip
            canvasCtx.lineTo(-arrowLength, arrowWidth);  // bottom right
            canvasCtx.lineTo(-arrowLength, -arrowWidth); // bottom left
            canvasCtx.closePath();

            canvasCtx.fillStyle = '#000000';
            canvasCtx.fill();
            
            canvasCtx.restore();
        }

        this.clientState.rendersSinceUpdate++;
    }

    onDestroyAnimation(eventData) {
        const x = eventData["x"];
        const y = eventData["y"];
        if (x == null || y == null) {
            return null;
        }
        const startingTime = performance.now();

        return (ctx, timestamp) => {
            const timeSeconds = (timestamp - startingTime) / 1000.0;
            const ratio = timeSeconds / Constants.ARROW_DESTROY_ANIMATION_TIME;
            
            // Draw expanding 'X'
            const size = ratio * 12;  // Max size of 12px
            ctx.beginPath();
            ctx.strokeStyle = '#ff6666';
            ctx.lineWidth = 2;
            
            ctx.moveTo(x - size, y - size);
            ctx.lineTo(x + size, y + size);
            ctx.moveTo(x + size, y - size);
            ctx.lineTo(x - size, y + size);
            
            ctx.stroke();

            return ratio < 1.0;  // Return false when animation is complete
        };
    }
}
